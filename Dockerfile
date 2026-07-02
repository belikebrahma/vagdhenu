FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU first to avoid downloading CUDA binaries
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torchcodec || true

# Clone NVIDIA BigVGAN repo
RUN git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git BigVGAN

# Add BigVGAN to PYTHONPATH
ENV PYTHONPATH="/app/BigVGAN:${PYTHONPATH}"

# Copy the rest of the application
COPY . .

# Create output directories
RUN mkdir -p out

# Expose API port
EXPOSE 8000

# Run FastAPI API microservice
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
