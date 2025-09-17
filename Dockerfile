# Use a lightweight Python base image
FROM python:3.11-slim

# Install system dependencies required for pdf2image and PyMuPDF
RUN apt-get update && apt-get install -y \
    poppler-utils \
    build-essential \
    libpoppler-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install Python dependencies directly (no requirements.txt)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pdf2image \
    PyMuPDF

# Expose FastAPI default port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
