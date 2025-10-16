# Server Setup and Deployment Guide

Complete guide for setting up and deploying the music analysis Docker image on Ubuntu 24.04 LTS.

## 1. Docker Setup

```bash
# Install dependencies
sudo apt update
sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg lsb-release

# Add Docker repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
sudo docker run hello-world
```

## 2. Download Repository

```bash
git clone https://github.com/...
cd mvsep-mdx23-docker
git checkout v2.6.1
```

## 3. UV Installation and Requirements

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version

# Create virtual environment
uv venv .venv --python=python3.12
source .venv/bin/activate

# Install requirements for download script
uv pip install numpy tqdm aiohttp huggingface_hub setuptools
```

## 4. Download Models

```bash
uv run utils/download_models.py
```

This downloads all required models including BS RoFormer, MDX23C, Demucs variants, and Whisper large-v3.

## 5. Docker Build

```bash
# Login to Docker Hub
docker login

# Build image
sudo docker build -t music-analysis .
```

## 6. Tag and Push

```bash
# login 
docker login

# Tag image
docker tag music-analysis aivividup/music-analysis:1.0.0

# Push to Docker Hub
docker push aivividup/music-analysis:1.0.0
```

---

**Image**: `aivividup/music-analysis:1.0.0`
**Base**: nvidia/cuda:12.4.0-runtime-ubuntu22.04
**Python**: 3.12
