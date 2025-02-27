# Base image with PyTorch
FROM pytorch/pytorch:2.1.0

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with MPS support
RUN pip install --no-cache-dir torch torchvision torchaudio

# Copy project files
COPY . /app
WORKDIR /app

# Set environment variables
ENV PYTORCH_ENABLE_MPS_FALLBACK=1 FORCE_DEVICE="mps"

# Run the script
ENTRYPOINT ["python", "gen.py"]
