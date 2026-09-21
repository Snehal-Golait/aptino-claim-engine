FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source and data files
COPY . .

# Ingestion is pre-built in storage/ (Chroma + BM25), but build if missing on startup
EXPOSE 8000 8501

# Default startup script runs both FastAPI (8000) and Streamlit (8501)
RUN chmod +x start.sh || true

CMD ["bash", "start.sh"]
