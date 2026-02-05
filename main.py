from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
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
from typing import List
import io
import img2pdf

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

# صيغ الصور المدعومة للاستخراج المباشر
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

def is_pdf(filename: str) -> bool:
    return filename.lower().endswith(".pdf")

def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS

# ========== الـ API ==========
@app.post("/extract-images")
async def extract_images(file: UploadFile = File(...), request: Request = None):
    fn = (file.filename or "").lower()
    if not is_pdf(file.filename) and not is_image_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="الملف يجب أن يكون بصيغة PDF أو صورة (jpg, png, gif, webp, bmp)",
        )

    # Temporary folder
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename or "upload")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extracted_images = []
        session_id = str(uuid.uuid4())
        output_folder = OUTPUT_BASE / session_id
        output_folder.mkdir(parents=True, exist_ok=True)

        if is_image_file(file.filename):
            # استخراج مباشر من ملف صورة
            ext = Path(file.filename).suffix.lower()
            if ext == ".jpg":
                ext = ".jpeg"
            single_path = output_folder / f"image_1{ext}"
            shutil.copy2(file_path, single_path)

            # دعم GIF متعدد الإطارات: استخراج كل الإطارات
            if ext == ".gif":
                gif = cv2.VideoCapture(str(single_path))
                frame_idx = 0
                while True:
                    ret, frame = gif.read()
                    if not ret:
                        break
                    frame_path = output_folder / f"frame_{frame_idx + 1}.png"
                    cv2.imwrite(str(frame_path), frame)
                    extracted_images.append(f"/static/images/{session_id}/{frame_path.name}")
                    frame_idx += 1
                gif.release()
                single_path.unlink(missing_ok=True)
                if frame_idx == 0:
                    # لم يُقرأ أي إطار (ملف تالف أو غير مدعوم)
                    extracted_images = []
            else:
                # ملف صورة واحد: نرجعه دائماً كمستخرج (بدون اشتراط وجود وجه)
                extracted_images.append(f"/static/images/{session_id}/{single_path.name}")
        else:
            # معالجة PDF كما سابقاً
            pdf_path = file_path
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
                    extracted_images.append(f"/static/images/{session_id}/{image_filename.name}")

            pdf_document.close()

            poppler_path = os.getenv("POPPLER_PATH", None)
            images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
            for i, image in enumerate(images):
                image_filename = output_folder / f"page_{i+1}.png"
                image.save(image_filename, "PNG")
                extracted_images.append(f"/static/images/{session_id}/{image_filename.name}")

        if not extracted_images:
            raise HTTPException(status_code=404, detail="لم يتم العثور على أي صور في الملف")

        base_url = str(request.base_url).rstrip("/")
        full_links = [f"{base_url}{url}" for url in extracted_images]

        return JSONResponse(content={"image_urls": full_links})

    except HTTPException:
        raise
    except Exception as e:
        logging.error("Error processing file: %s", str(e))
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة الملف: {str(e)}")

    finally:
        file.file.close()


@app.post("/convert")
async def convert(images: List[UploadFile] = File(...)):
    """
    استقبال عدة صور وتحويلها إلى ملف PDF واحد.
    يعادل راوت Flask التالي:
    /convert (POST) مع حقل form-data باسم images (multiple files)
    """
    if not images:
        raise HTTPException(status_code=400, detail="No files uploaded")

    img_list: List[bytes] = []
    for f in images:
        if f.filename:
            contents = await f.read()
            if contents:
                img_list.append(contents)

    if not img_list:
        raise HTTPException(status_code=400, detail="No images selected")

    try:
        # Convert images bytes إلى PDF bytes
        pdf_bytes = img2pdf.convert(img_list)

        pdf_io = io.BytesIO(pdf_bytes)
        pdf_io.seek(0)

        headers = {
            "Content-Disposition": 'attachment; filename="converted.pdf"'
        }

        return StreamingResponse(
            pdf_io,
            media_type="application/pdf",
            headers=headers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
