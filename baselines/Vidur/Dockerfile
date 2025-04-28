# Use the NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python and upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip

# Install TensorRT-LLM from PyPI
RUN pip install --no-cache-dir virtualenv
# RUN python -m pip install torch transformers datasets accelerate peft

# Set up the working directory
WORKDIR /workspace

# Set the default command
CMD ["/bin/bash"]
