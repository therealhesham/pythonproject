# Use a lightweight Python base image
FROM python:3.11-slim-buster

# Install system dependencies required for pdf2image and PyMuPDF
# poppler-utils: for pdf2image
# libpoppler-dev: for PyMuPDF to work correctly with Poppler (though PyMuPDF is often self-contained for rendering, this is good for robustness)
# build-essential: generally useful for compiling extensions, though might not be strictly needed for PyMuPDF itself
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libpoppler-dev \
    # build-essential # ممكن متحتجهاش لو مفيش C extensions بتتبنى
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy your application files into the container
# We copy all files (including your main.py and potentially other modules)
COPY . .

# Install Python dependencies directly (you can also use a requirements.txt file)
# uvicorn[standard] includes httptools and websockets for better performance
# PyMuPDF is the correct package for 'fitz'
# Pillow is usually a dependency for image manipulation libraries like pdf2image
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pdf2image \
    PyMuPDF \
    Pillow

# Expose the port your FastAPI application will listen on
EXPOSE 8000

# Command to run your FastAPI application using uvicorn
# "main:app" assumes your FastAPI application instance is named 'app'
# and is defined in a file named 'main.py'
# --host 0.0.0.0 makes the server accessible from outside the container
# --port 8000 specifies the port
# --log-level debug will show detailed logs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]