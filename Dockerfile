FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

## Copy requirements file into container
#WORKDIR /app
#COPY requirements.txt .
#
## Python deps (from file)
#RUN pip install --no-cache-dir -r requirements.txt
#
## Detectron2 (install after base deps)
#RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
