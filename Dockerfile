# Use a lightweight Python base image with a recent Debian release
FROM python:3.11-slim-bullseye

# Set environment variable to prevent apt-get from prompting for user input
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for pdf2image, PyMuPDF, and OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
        libpoppler-dev \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy your application files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pdf2image \
    PyMuPDF \
    Pillow \
    python-multipart \
    opencv-python-headless \
    apscheduler

# Make sure static folder exists
RUN mkdir -p /app/static/images

# Persist static files (optional but recommended)
VOLUME ["/app/static"]

# Expose the port your FastAPI application will listen on
EXPOSE 8000

# Command to run your FastAPI application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
