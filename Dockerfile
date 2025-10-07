# Specify the CUDA and Ubuntu versions as build arguments.
# You can later override CUDA_VERSION with any value between 12.2.0 and 12.9.0
ARG CUDA_VERSION=12.4.0
ARG UBUNTU_VERSION=22.04

# Use the CUDA runtime image (if you need extra development tools, replace "runtime" with "devel")
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-distutils python3-pip \
        git ffmpeg curl libsndfile1 libcudnn8 libcudnn8-dev \
        nvidia-container-toolkit \
        libfftw3-dev libfftw3-3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Install Python dependencies
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create and copy model files
COPY models/ /models/

# Copy handler code
COPY src/handler.py .
COPY src/inference.py .
COPY src/chords_tempo.py .
COPY src/modules/ ./modules/

# Pre-download nltk punkt tokenizer to image
RUN mkdir -p /usr/share/nltk_data && \
    python3 -c "import nltk; nltk.download('punkt', download_dir='/usr/share/nltk_data')" && \
    python3 -c "import nltk; nltk.data.path.append('/usr/share/nltk_data')"

# Pre-download WhisperX large-v3 model
RUN mkdir -p /models/whisper && \
    python3 -c "import whisperx; whisperx.load_model('large-v3', 'cpu', compute_type='int8', download_root='/models/whisper')" && \
    echo "WhisperX large-v3 model downloaded successfully"

# Test CUDA availability
RUN python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Set default command to run handler
CMD [ "python3", "-u", "/handler.py" ]