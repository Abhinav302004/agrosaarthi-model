FROM --platform=linux/amd64 python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements.txt

# Set working directory
WORKDIR /app

# Copy entire backend folder contents to /app
COPY . /app/

# Expose the port for FastAPI
EXPOSE 8000

# Start the FastAPI app (corrected from main:app to app:app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
