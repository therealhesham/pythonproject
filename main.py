from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import os
import shutil
from pathlib import Path
import tempfile
import logging
import cv2
import uuid
import time
from apscheduler.schedulers.background import BackgroundScheduler
import numpy as np

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_BASE = Path("static/images")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

EXPIRY_SECONDS = 24 * 60 * 60

# ========== دالة قص وحفظ الوجوه ==========
def extract_faces_and_save(image_path: str, output_folder: Path, base_filename: str) -> list:
    """
    تقوم هذه الدالة بقص الوجوه من الصورة وحفظها كملفات منفصلة.
    ترجع قائمة بأسماء الملفات الجديدة.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # تعديل البارامترات لتقليل النتائج الخاطئة (minNeighbors=6)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
    
    saved_faces_paths = []
    
    if len(faces) > 0:
        height, width, _ = img.shape
        
        # هامش إضافي حول الوجه (Padding) عشان الصورة متكونش مخنوقة
        padding = 10 

        for i, (x, y, w, h) in enumerate(faces):
            # التأكد من أن الإحداثيات داخل حدود الصورة
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)
            
            # عملية القص (Cropping)
            face_img = img[y1:y2, x1:x2]
            
            # حفظ الوجه في ملف جديد
            face_filename = f"face_{base_filename}_{i+1}.jpg"
            save_path = output_folder / face_filename
            cv2.imwrite(str(save_path), face_img)
            
            # إضافة المسار النسبي للقائمة
            saved_faces_paths.append(face_filename)

    return saved_faces_paths

# ========== دالة مسح الملفات القديمة ==========
def cleanup_old_sessions():
    now = time.time()
    for session_dir in OUTPUT_BASE.iterdir():
        if session_dir.is_dir():
            created_at = session_dir.stat().st_mtime
            if now - created_at > EXPIRY_SECONDS:
                shutil.rmtree(session_dir, ignore_errors=True)

scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_sessions, "interval", hours=1)
scheduler.start()

# ========== الـ API ==========
@app.post("/extract-faces")  # غيرت الاسم ليكون أوضح
async def extract_faces(file: UploadFile = File(...), request: Request = None):
    filename = file.filename.lower()
    is_pdf = filename.endswith(".pdf")
    is_image = filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))

    if not (is_pdf or is_image):
        raise HTTPException(status_code=400, detail="الملف غير مدعوم")

    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        final_face_urls = []
        session_id = str(uuid.uuid4())
        output_folder = OUTPUT_BASE / session_id
        output_folder.mkdir(parents=True, exist_ok=True)

        # --- معالجة الصور المباشرة ---
        if is_image:
            faces = extract_faces_and_save(input_path, output_folder, "uploaded")
            for face in faces:
                final_face_urls.append(f"/static/images/{session_id}/{face}")

        # --- معالجة ملفات PDF ---
        elif is_pdf:
            # 1. تحويل صفحات PDF لصور عالية الدقة (أفضل طريقة لضمان التقاط كل شيء)
            poppler_path = os.getenv("POPPLER_PATH", None)
            try:
                images_from_pdf = convert_from_path(input_path, dpi=200, poppler_path=poppler_path)
            except Exception as e:
                # Fallback: لو فشل poppler نستخدم fitz للصور المدمجة
                 logging.warning(f"Poppler failed, falling back to extraction: {e}")
                 images_from_pdf = []

            # معالجة الصور المحولة من الصفحات
            for i, image in enumerate(images_from_pdf):
                # نحفظ صفحة الـ PDF كصورة مؤقتة عشان OpenCV يقرأها
                page_temp_path = os.path.join(temp_dir, f"page_{i}.jpg")
                image.save(page_temp_path, "JPEG")
                
                # نستخرج الوجوه من هذه الصفحة
                faces = extract_faces_and_save(page_temp_path, output_folder, f"page_{i}")
                for face in faces:
                    final_face_urls.append(f"/static/images/{session_id}/{face}")

            # 2. (اختياري) استخراج الصور المدمجة (Embedded) لو Poppler مجبش نتيجة كويسة
            # يمكن إزالتها لتسريع الكود إذا كانت طريقة poppler كافية
            if not final_face_urls:
                 pdf_document = fitz.open(input_path)
                 for page_num in range(len(pdf_document)):
                    for img in pdf_document[page_num].get_images(full=True):
                        xref = img[0]
                        base = pdf_document.extract_image(xref)
                        temp_img_path = os.path.join(temp_dir, f"embed_{xref}.{base['ext']}")
                        with open(temp_img_path, "wb") as f:
                            f.write(base["image"])
                        
                        faces = extract_faces_and_save(temp_img_path, output_folder, f"emb_{xref}")
                        for face in faces:
                            final_face_urls.append(f"/static/images/{session_id}/{face}")
                 pdf_document.close()

        if not final_face_urls:
            shutil.rmtree(output_folder, ignore_errors=True) # حذف المجلد الفارغ
            raise HTTPException(status_code=404, detail="لم يتم العثور على أي وجوه لقصها")

        base_url = str(request.base_url).rstrip("/")
        full_links = [f"{base_url}{url}" for url in final_face_urls]

        return JSONResponse(content={"face_urls": full_links, "count": len(full_links)})

    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
