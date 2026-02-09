# Multi-stage build for SF Muni Status API
# Uses security best practices: minimal image, non-root user, health checks
# Build with Cloud Build: gcloud builds submit --tag IMAGE_NAME

# Stage 1: Build stage - install dependencies
FROM python:3.13-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage - minimal production image
FROM python:3.13-slim-bookworm

# Install runtime dependencies
# - curl: health checks and downloading tessdata
# - tesseract-ocr: OCR engine for train ID detection
# - libgl1: OpenCV headless runtime
# Note: We download tessdata_best instead of using tesseract-ocr-eng package
# because the "best" model has higher accuracy than the default "fast" model
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -L -o /usr/share/tesseract-ocr/5/tessdata/eng.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set up non-root user for security
RUN groupadd -r muni && \
    useradd -r -g muni -d /app -s /sbin/nologin muni

# Set working directory
WORKDIR /app

# Copy shared library
COPY --chown=muni:muni lib/ ./lib/

# Copy detection scripts (required by lib/detection.py)
COPY --chown=muni:muni scripts/ ./scripts/

# Copy API application files to api/ subdirectory to preserve path structure
COPY --chown=muni:muni api/ ./api/

# Create directories for runtime data with proper permissions
RUN mkdir -p /app/artifacts/runtime/downloads /app/artifacts/runtime/cache && \
    chown -R muni:muni /app

# Switch to non-root user
USER muni

# Set Python path to use virtual environment and find lib/
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:${PYTHONPATH}"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    ENABLE_FALLBACK=false \
    CLOUD_RUN=true \
    GCS_BUCKET=munimetro-cache \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check - verify API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run gunicorn with proper worker configuration
# - Multiple workers now possible (no heavy ML model)
# - Graceful timeout for proper shutdown
CMD ["gunicorn", \
    "api.api:app", \
    "--bind", "0.0.0.0:8000", \
    "--workers", "2", \
    "--timeout", "60", \
    "--graceful-timeout", "30", \
    "--log-level", "info", \
    "--access-logfile", "-"]
