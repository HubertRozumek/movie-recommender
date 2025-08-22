# Multi-stage build for MovieLens Recommendation System

# Stage 1: Base image with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation
FROM base as dependencies

WORKDIR /tmp

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 3: Application
FROM base as runtime

# Create non-root user
RUN useradd -m -u 1000 mluser && \
    mkdir -p /app /app/data /app/results /app/logs /app/experiments && \
    chown -R mluser:mluser /app

# Copy Python dependencies from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=mluser:mluser . .

# Install the package
RUN pip install -e .

# Switch to non-root user
USER mluser

# Expose ports
EXPOSE 7860 5000 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default command
CMD ["python", "app/main.py"]