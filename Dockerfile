FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

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