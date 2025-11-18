# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU acceleration, optional)
- 8GB+ RAM (16GB+ recommended)
- GPU with 4GB+ VRAM (for faster processing)

## Installation Methods

### 1. From Source (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/alexv879/Security-Camera-Face-Image-Enhancement.git
cd Security-Camera-Face-Image-Enhancement

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 2. Using pip (When published to PyPI)

```bash
pip install face-enhancement-ai
```

### 3. Using Docker

```bash
# Pull image (when published)
docker pull face-enhancement-ai:latest

# Or build locally
git clone https://github.com/alexv879/Security-Camera-Face-Image-Enhancement.git
cd Security-Camera-Face-Image-Enhancement
docker build -t face-enhancement-ai .
```

## GPU Setup

### CUDA Installation

1. **Check CUDA availability:**
   ```bash
   nvidia-smi
   ```

2. **Install CUDA Toolkit (if not installed):**
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Follow installation instructions for your OS

3. **Install PyTorch with CUDA:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Verify GPU Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## CPU-Only Installation

If you don't have a GPU or want CPU-only installation:

```bash
# Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
pip install -e .
```

## Model Weights Download

Models are automatically downloaded on first use. To manually download:

```bash
mkdir -p models/weights

# GFPGAN v1.4
wget -P models/weights https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# Real-ESRGAN x2plus
wget -P models/weights https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

# Detection model (RetinaFace) - downloaded automatically by facexlib
```

## Verify Installation

```bash
# Check CLI
face-enhance --version
face-enhance info

# Check Python API
python -c "from face_enhancement import EnhancementPipeline; print('OK')"

# Run tests
pytest tests/
```

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'face_enhancement'`

**Solution:**
```bash
pip install -e .
```

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce upscale factor: `--upscale 2` instead of `--upscale 4`
- Process smaller batches
- Use CPU: `--device cpu`
- Close other GPU applications

### Model Download Fails

**Problem:** Model weights download timeout

**Solution:**
```bash
# Download manually using browser or wget
# Place in models/weights/
```

### OpenCV Issues

**Problem:** `ImportError: libGL.so.1: cannot open shared object file`

**Solution (Linux):**
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### macOS M1/M2 Installation

```bash
# Use conda for M1/M2 Macs
conda create -n face-enhance python=3.10
conda activate face-enhance

# Install PyTorch for M1/M2
conda install pytorch torchvision -c pytorch

# Install other dependencies
pip install -r requirements.txt
pip install -e .
```

## Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run code formatters
make format

# Run tests
make test
```

## Docker Installation

### Build Image

```bash
docker build -t face-enhancement-ai .
```

### Run Container

```bash
# With GPU
docker run --gpus all -p 8000:8000 face-enhancement-ai

# CPU only
docker run -p 8000:8000 face-enhancement-ai

# With volume mount
docker run --gpus all -v $(pwd)/data:/app/data face-enhancement-ai \
  face-enhance enhance /app/data/input.jpg
```

### Docker Compose

```bash
# Start API server with GPU
docker-compose up face-enhancement-api

# Start API server with CPU
docker-compose --profile cpu-only up face-enhancement-cpu
```

## Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-enhancement
spec:
  replicas: 2
  selector:
    matchLabels:
      app: face-enhancement
  template:
    metadata:
      labels:
        app: face-enhancement
    spec:
      containers:
      - name: api
        image: face-enhancement-ai:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
```

```bash
kubectl apply -f deployment.yaml
```

## System Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB
- OS: Ubuntu 18.04+, Windows 10+, macOS 10.15+

### Recommended Requirements
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with 6GB+ VRAM
- Storage: 20GB+ SSD
- OS: Ubuntu 20.04+, Windows 11, macOS 12+

## Next Steps

After installation:
1. Read the [Quick Start Guide](../README.md#quick-start)
2. Check [API Documentation](API.md)
3. Try [Example Scripts](../examples/example_usage.py)
4. Configure settings in `config/default_config.yaml`

## Support

If you encounter issues:
- Check [Troubleshooting](#troubleshooting) section
- Search [GitHub Issues](https://github.com/alexv879/Security-Camera-Face-Image-Enhancement/issues)
- Create a new issue with detailed error messages
