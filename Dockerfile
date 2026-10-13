# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python Backend Runtime
FROM python:3.10-slim

WORKDIR /app

# Install system libraries for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
COPY ml/requirements.txt /app/ml/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt && \
    pip install --no-cache-dir -r /app/ml/requirements.txt

# Copy backend & ML modules
COPY backend /app/backend
COPY ml /app/ml

# Copy built frontend dist from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Set up non-root user (Hugging Face Spaces requirement: UID 1000)
RUN useradd -m -u 1000 user && \
    mkdir -p /app/backend/storage && \
    chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app/backend:/app \
    PORT=7860

WORKDIR /app/backend

# Hugging Face default port is 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
