FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install Tesseract OCR and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p /app/data

# Create startup script
RUN echo '#!/bin/bash\n\
# Start the FastAPI application\nuvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}' > /app/start.sh \
    && chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["/app/start.sh"]
