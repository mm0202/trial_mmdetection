FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install OpenMIM
RUN pip install --no-cache-dir -U openmim

# Install MMEngine
RUN mim install mmengine

# Install MMCV (GPU version)
# RUN mim install "mmcv>=2.0.0rc4,<2.2.0"
RUN pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# Install MMDetection
RUN mim install mmdet

# Install ipykernel for Jupyter support
RUN pip install --no-cache-dir ipykernel
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir tensorboard

RUN apt-get update && apt-get install -y unzip