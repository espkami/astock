FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (overridden by volume mount in dev)
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Data directory
RUN mkdir -p /app/data

EXPOSE 8000 5678

# Entrypoint: debug mode when DEBUG=1, else production (no --reload)
CMD ["sh", "-c", "\
  if [ \"$DEBUG\" = '1' ]; then \
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
      -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload; \
  else \
    uvicorn backend.main:app --host 0.0.0.0 --port 8000; \
  fi"]
