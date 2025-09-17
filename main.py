from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import os
import shutil
from pathlib import Path
import tempfile
import logging
import zipfile

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

@app.post("/extract-images/")
async def extract_images(file: UploadFile = File(...)):
    # Validate PDF file
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="الملف يجب أن يكون بصيغة PDF")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    output_folder = os.path.join(temp_dir, "output_images")
    os.makedirs(output_folder, exist_ok=True)

    # Save uploaded PDF
    pdf_path = os.path.join(temp_dir, file.filename)
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extracted_images = []

        # 1. Extract embedded images using PyMuPDF
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"{output_folder}/embedded_image_page{page_num + 1}_{img_index + 1}.{image_ext}"
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)
                extracted_images.append(image_filename)

        pdf_document.close()

        # 2. Convert PDF pages to images using pdf2image
        poppler_path = os.getenv("POPPLER_PATH", None)  # None = Linux (Docker), set if Windows
        images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
        for i, image in enumerate(images):
            image_filename = f"{output_folder}/page_{i + 1}.png"
            image.save(image_filename, "PNG")
            extracted_images.append(image_filename)

        # 3. Create a ZIP file containing all extracted images
        zip_path = os.path.join(temp_dir, "extracted_images.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for img_file in extracted_images:
                zipf.write(img_file, arcname=os.path.basename(img_file))

        # Return ZIP file as response
        return FileResponse(
            path=zip_path,
            filename="extracted_images.zip",
            media_type="application/zip"
        )

    except Exception as e:
        logging.error("Error processing file: %s", str(e))
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة الملف: {str(e)}")

    finally:
        file.file.close()
        # ملاحظــة: بنسيب temp_dir لأن FileResponse محتاج الملف يفضل موجود
        # ممكن نعمل background task عشان نمسحه بعد ما يتحمل
