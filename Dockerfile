# Use a lightweight Python base image with a more recent Debian release.
# 'slim-bullseye' or 'slim-bookworm' are good choices.
# Let's try 'slim-bullseye' first.
FROM python:3.11-slim-bullseye

# Set environment variable to prevent apt-get from prompting for user input
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for pdf2image and PyMuPDF
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
        libpoppler-dev \
        # build-essential # يمكنك إلغاء التعليق إذا لزم الأمر
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
    Pillow

# Expose the port your FastAPI application will listen on
EXPOSE 8000

# Command to run your FastAPI application with uvicorn
CMD ["sleep", "infinity"]