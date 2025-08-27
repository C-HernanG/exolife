# ExoLife Docker Container
# Multi-stage build for optimized production image

# Build stage: Install dependencies
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY package/ ./package/

# Install Python dependencies from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .[dev,jupyter]

# Production stage: Create lightweight runtime image
FROM python:3.11-slim

# Set metadata
LABEL maintainer="Carlos Hernán Guirao <carlos.hernangui@gmail.com>"
LABEL description="ExoLife: Predicting Exoplanet Habitability to Support Astrobiological Discovery"
LABEL version="0.1.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.local/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    # Install NVIDIA CUDA Toolkit dependencies
    # https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#os-integration
    # These packages are often needed by CUDA-dependent applications and might not be included in the base image.
    # Adding them here to ensure a smoother GPU environment setup.
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --user-group exolife && \
    mkdir -p /app/data && \
    chown -R exolife:exolife /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=exolife:exolife package/ ./package/
COPY --chown=exolife:exolife config/ ./config/
COPY --chown=exolife:exolife scripts/ ./scripts/
COPY --chown=exolife:exolife pyproject.toml README.md ./

# Install the package in the final image
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p /app/data/{raw,interim,processed} && \
    chown -R exolife:exolife /app/data

# Switch to non-root user
USER exolife

# Create volume mount points
VOLUME ["/app/data", "/app/notebooks", "/app/results"]

# Expose Jupyter port (if needed)
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD exolife --help || exit 1

# Set default command
CMD ["bash"]
