# Multi-stage build for SF Muni Status API
# Uses security best practices: minimal image, non-root user, health checks
# Build from project root: docker build -f api/Dockerfile .

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

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set up non-root user for security
RUN groupadd -r muni && \
    useradd -r -g muni -d /app -s /sbin/nologin muni

# Set working directory
WORKDIR /app

# Copy shared library
COPY --chown=muni:muni lib/ ./lib/

# Copy API application files
COPY --chown=muni:muni api/api.py .
COPY --chown=muni:muni api/check_status.py .
COPY --chown=muni:muni api/predict_status.py .
COPY --chown=muni:muni api/index.html .

# Copy download script (used by check_status.py)
COPY --chown=muni:muni training/download_muni_image.py .

# Create directories for runtime data with proper permissions
RUN mkdir -p /app/data/muni_snapshots /app/data/cache /app/models/trained_model && \
    chown -R muni:muni /app

# Switch to non-root user
USER muni

# Set Python path to use virtual environment and find lib/
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:${PYTHONPATH}"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    ENABLE_FALLBACK=false \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check - verify API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run gunicorn with proper worker configuration
# - Single worker to avoid memory issues with ML model
# - Graceful timeout for proper shutdown
# - Access logging disabled for performance (enable with --access-logfile -)
CMD ["gunicorn", \
    "api:app", \
    "--bind", "0.0.0.0:8000", \
    "--workers", "1", \
    "--timeout", "120", \
    "--graceful-timeout", "30", \
    "--log-level", "info", \
    "--access-logfile", "-"]
