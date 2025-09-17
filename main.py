from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import os
import shutil
from pathlib import Path
import tempfile
import logging
import cv2  # OpenCV
import uuid
import time
from apscheduler.schedulers.background import BackgroundScheduler

# إعداد FastAPI
app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

# إضافة CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# مجلد ثابت لتخزين الصور
OUTPUT_BASE = Path("static/images")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# مدة الاحتفاظ بالصور (24 ساعة = 86400 ثانية)
EXPIRY_SECONDS = 24 * 60 * 60

# ========== دالة كشف الوجوه ==========
def contains_face(image_path: str) -> bool:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

# ========== دالة مسح الملفات القديمة ==========
def cleanup_old_sessions():
    now = time.time()
    for session_dir in OUTPUT_BASE.iterdir():
        if session_dir.is_dir():
            created_at = session_dir.stat().st_mtime
            if now - created_at > EXPIRY_SECONDS:
                logging.info(f"🗑️ حذف المجلد: {session_dir}")
                shutil.rmtree(session_dir, ignore_errors=True)

# تشغيل الجدولة كل ساعة
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_sessions, "interval", hours=1)
scheduler.start()

# ========== الـ API ==========
@app.post("/extract-images/")
async def extract_images(file: UploadFile = File(...), request: Request = None):
    # Validate PDF file
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="الملف يجب أن يكون بصيغة PDF")

    # Temporary folder
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, file.filename)
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extracted_images = []

        # مجلد خاص لكل رفع
        session_id = str(uuid.uuid4())
        output_folder = OUTPUT_BASE / session_id
        output_folder.mkdir(parents=True, exist_ok=True)

        # 1. استخراج الصور المدمجة داخل PDF
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = output_folder / f"embedded_page{page_num+1}_{img_index+1}.{image_ext}"
                with open(image_filename, "wb") as f:
                    f.write(image_bytes)

                if contains_face(str(image_filename)):
                    extracted_images.append(f"/static/images/{session_id}/{image_filename.name}")
                else:
                    image_filename.unlink()

        pdf_document.close()

        # 2. تحويل الصفحات لصور
        poppler_path = os.getenv("POPPLER_PATH", None)
        images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
        for i, image in enumerate(images):
            image_filename = output_folder / f"page_{i+1}.png"
            image.save(image_filename, "PNG")

            if contains_face(str(image_filename)):
                extracted_images.append(f"/static/images/{session_id}/{image_filename.name}")
            else:
                image_filename.unlink()

        if not extracted_images:
            raise HTTPException(status_code=404, detail="مفيش صور فيها وش في الملف ده")

        # توليد الروابط كاملة بالـ host
        base_url = str(request.base_url).rstrip("/")
        full_links = [f"{base_url}{url}" for url in extracted_images]

        return JSONResponse(content={"image_urls": full_links})

    except Exception as e:
        logging.error("Error processing file: %s", str(e))
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة الملف: {str(e)}")

    finally:
        file.file.close()