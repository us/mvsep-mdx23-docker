# Specify the CUDA and Ubuntu versions as build arguments.
# You can later override CUDA_VERSION with any value between 12.2.0 and 12.9.0
ARG CUDA_VERSION=12.2.0
ARG UBUNTU_VERSION=22.04

# Use the CUDA runtime image (if you need extra development tools, replace "runtime" with "devel")
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    git \
    ffmpeg \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create models directory and copy pre-downloaded models
COPY models/* /workspace/models/

# Set default command
CMD ["python3", "-u", "inference.py"] 