FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Create a non-root user to run the application
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Create startup script
RUN echo '#!/bin/bash\n\
# Start the FastAPI application\nuvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}' > /app/start.sh \
    && chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["/app/start.sh"]
