# syntax=docker/dockerfile:1
# Using a base Ubuntu 22.04 image with CUDA 12.1 for PyTorch compatibility
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies including python3.11 and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gcc \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy and install Python requirements.
COPY app_backend/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    CMAKE_ARGS="-DGGML_OPENMP=OFF" pip install --default-timeout=1000 -r requirements.txt

# Copy entire project (model artifacts are excluded via .dockerignore
# and mounted as volumes at runtime anyway)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
