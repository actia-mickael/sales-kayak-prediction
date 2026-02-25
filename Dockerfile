# =============================================================================
# 🛶 Sales Kayak Prediction Dashboard - Dockerfile
# =============================================================================

FROM python:3.10-slim

# Metadata
LABEL maintainer="Mickael - ACT-IA"
LABEL description="Sales Kayak Customer Prediction Dashboard"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY model_utils.py .
COPY app.py .

# Create directories
RUN mkdir -p data models

# Copy data (optional - can be mounted as volume)
COPY data/ data/

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860')" || exit 1

# Run application
CMD ["python", "app.py"]
