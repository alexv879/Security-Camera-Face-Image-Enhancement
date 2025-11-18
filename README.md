# Face Enhancement AI 🎯

State-of-the-art face image enhancement system specifically designed for security camera footage. Powered by advanced deep learning models including **GFPGAN**, **CodeFormer**, **Real-ESRGAN**, **Stable Diffusion**, and **Vision Transformers**.

## 🌟 2025+ Cutting-Edge Features

**NEW!** This system now includes the most advanced AI technologies:

- 🎨 **Diffusion Models** - Stable Diffusion & ControlNet for photorealistic enhancement
- 🔮 **Vision Transformers** - Swin Transformer & ViT for global context understanding
- 🖥️ **Interactive Web UI** - Beautiful Gradio interface with 4 enhancement modes
- 📦 **MLOps Platform** - Model versioning, A/B testing, deployment tracking, drift detection
- 🌐 **3D Face Reconstruction** - 3D morphable models, novel view synthesis, pose estimation
- 📱 **Mobile Deployment** - TensorFlow Lite support for iOS, Android, Raspberry Pi, Jetson
- 🔒 **Privacy-Preserving ML** - Differential privacy, federated learning, homomorphic encryption

**[See Full 2025+ Features Documentation →](docs/2025_FEATURES.md)**

## ✨ Features

### Core Enhancement
- 🔍 **Advanced Face Detection** - RetinaFace with facial landmark detection
- 🎨 **Multiple Enhancement Models** - GFPGAN and CodeFormer support
- 🌙 **Low-Light Enhancement** - CLAHE-based brightness optimization
- 🔊 **Noise Reduction** - Bilateral filtering and NLM denoising
- 📈 **Super-Resolution** - Up to 4x upscaling with Real-ESRGAN
- 🎭 **Model Ensemble** - Combine multiple models for superior results

### Advanced Preprocessing
- 🌈 **HDR Tone Mapping** - Reinhard, Drago, and Mantiuk algorithms
- 🎨 **Advanced Color Correction** - Gray world, Max RGB, Multi-scale Retinex
- 🌫️ **Fog/Haze Removal** - Dark channel prior algorithm
- ✨ **Perceptual Sharpening** - Unsharp masking in perceptual color space
- 🔆 **Local Adaptive Enhancement** - Varying lighting condition handling

### Production Features
- 🌐 **REST API** - FastAPI-based web service with OpenAPI docs
- 🎯 **CLI Tool** - Rich command-line interface with progress bars
- 📊 **Prometheus Metrics** - Real-time monitoring and observability
- 📈 **Quality Assessment** - PSNR, SSIM, sharpness, colorfulness metrics
- ⚡ **Caching Layer** - File-based and Redis caching support
- 🚀 **Model Optimization** - ONNX export and quantization
- 🎬 **Video Support** - Extract and enhance video frames
- ⚡ **Batch Processing** - Efficient multi-image processing
- 🐳 **Docker Support** - Production-ready containerized deployment
- 🧪 **Comprehensive Tests** - Unit and integration test coverage

### Advanced Capabilities (NEW!)
- 🎭 **Progressive Enhancement** - Multi-stage enhancement for extreme quality
- 📺 **Real-Time Streaming** - Live video stream enhancement with adaptive quality
- 🎨 **Visualization Tools** - Attention maps, quality charts, difference maps
- 🧠 **Perceptual Loss** - VGG-based perceptual quality metrics
- 🎯 **Face Fidelity** - Preserve facial identity while enhancing
- 📊 **Guided Enhancement** - Reference-based style transfer
- 🔍 **Explainability** - Understand what the model is doing
- ⚡ **Temporal Smoothing** - Smooth enhancement across video frames

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/alexv879/Security-Camera-Face-Image-Enhancement.git
cd Security-Camera-Face-Image-Enhancement

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

#### Command Line

```bash
# Basic enhancement
face-enhance enhance input.jpg --output enhanced.jpg

# Batch process a directory
face-enhance batch /path/to/images --output-dir /path/to/output

# Process video frames
face-enhance video surveillance.mp4 --interval 30 --max-frames 100

# Progressive multi-stage enhancement (NEW!)
python scripts/progressive_enhance.py input.jpg --stages 3 --visualize

# Real-time video stream enhancement (NEW!)
python scripts/stream_enhance.py --source 0 --fps 15 --output output.mp4

# Quality comparison with visualization (NEW!)
python scripts/quality_comparison.py input.jpg --model gfpgan --upscale 2

# Performance benchmarking
python scripts/benchmark.py --device cuda --runs 100

# View pipeline info
face-enhance info
```

#### Python API

```python
from face_enhancement.core.pipeline import EnhancementPipeline
from face_enhancement.config import AppConfig
import cv2

# Basic enhancement
config = AppConfig()
pipeline = EnhancementPipeline(config)
image = cv2.imread("low_quality_face.jpg")
enhanced, metadata = pipeline.process_image(image)

# Progressive enhancement (NEW!)
from face_enhancement.models.progressive_enhancer import ProgressiveEnhancer
progressive = ProgressiveEnhancer(stages=3, base_model="gfpgan")
enhanced, metadata = progressive.enhance(image, intermediate_outputs=True)

# Real-time streaming (NEW!)
from face_enhancement.streaming import VideoStreamEnhancer
stream = VideoStreamEnhancer(video_source="0", target_fps=15)
stream.run(display=True)

# Quality assessment with visualization (NEW!)
from face_enhancement.utils.quality_metrics import QualityAssessment
from face_enhancement.utils.visualization import EnhancementVisualizer

metrics = QualityAssessment.calculate_all_metrics(enhanced)
vis = EnhancementVisualizer.create_side_by_side(image, enhanced)
```

#### REST API

```bash
# Start API server
face-enhance-server

# Or with uvicorn
uvicorn face_enhancement.api.server:app --host 0.0.0.0 --port 8000
```

```bash
# Enhance image via API
curl -X POST "http://localhost:8000/enhance" \
  -F "file=@input.jpg" \
  -F "model=gfpgan" \
  -F "upscale=2" \
  --output enhanced.jpg

# Get detection results
curl -X POST "http://localhost:8000/detect" \
  -F "file=@input.jpg"
```

## 🎨 Enhancement Models

### GFPGAN (Default)
- **Best for**: General face restoration
- **Speed**: Fast
- **Quality**: High quality, balanced results
- **Use case**: Most security camera footage

### CodeFormer
- **Best for**: Heavily degraded images
- **Speed**: Moderate
- **Quality**: Excellent quality with fidelity control
- **Use case**: Very low-quality or heavily compressed images

## 📊 Pipeline Architecture

```
Input Image
    ↓
┌─────────────────────┐
│  Preprocessing      │
│  - Denoising        │
│  - CLAHE (low-light)│
│  - Auto contrast    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Face Detection     │
│  - RetinaFace       │
│  - Landmark points  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Face Enhancement   │
│  - GFPGAN/CodeFormer│
│  - Super-resolution │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Background Upscale │
│  - Real-ESRGAN      │
└──────────┬──────────┘
           ↓
    Enhanced Output
```

## ⚙️ Configuration

### YAML Configuration

Create `config.yaml`:

```yaml
device: cuda  # or 'cpu'
log_level: INFO

face_detection:
  model: retinaface
  confidence_threshold: 0.5
  nms_threshold: 0.4

enhancement:
  model: gfpgan  # or 'codeformer'
  upscale_factor: 2
  face_upsample: true
  bg_upsampler: realesrgan
  weight: 0.5  # CodeFormer fidelity (0=quality, 1=identity)

preprocessing:
  denoise: true
  auto_contrast: true
  enhance_brightness: true
  gamma_correction: 1.0
  sharpen: false

processing:
  batch_size: 1
  output_format: png
  output_quality: 95
  save_comparison: true
  save_detections: true
```

Use configuration:

```bash
face-enhance enhance input.jpg --config config.yaml
```

### Environment Variables

```bash
export FACE_ENHANCE_DEVICE=cuda
export FACE_ENHANCE_ENHANCEMENT__MODEL=gfpgan
export FACE_ENHANCE_ENHANCEMENT__UPSCALE_FACTOR=2
```

## 🔧 Advanced Usage

### Custom Preprocessing

```python
from face_enhancement.preprocessing.image_preprocessor import ImagePreprocessor

preprocessor = ImagePreprocessor(
    denoise=True,
    auto_contrast=True,
    enhance_brightness=True,
    gamma=1.2,  # Brighten image
    sharpen=True
)

processed_image = preprocessor.preprocess(image)
```

### Face Detection Only

```python
from face_enhancement.core.face_detector import FaceDetector

detector = FaceDetector(model="retinaface", device="cuda")
faces = detector.detect_faces(image, return_landmarks=True)

for bbox, confidence, landmarks in faces:
    print(f"Face at {bbox} with confidence {confidence}")
```

### Batch Processing with Progress

```python
from pathlib import Path

input_files = list(Path("input_dir").glob("*.jpg"))

results = pipeline.process_batch(
    input_files,
    output_dir=Path("output_dir"),
    progress_callback=lambda current, total: print(f"{current}/{total}")
)
```

## 📈 Performance

| Resolution | Model    | GPU (RTX 3090) | CPU (i7-10700K) |
|-----------|----------|----------------|-----------------|
| 512x512   | GFPGAN   | ~0.15s        | ~2.5s          |
| 512x512   | CodeFormer| ~0.25s       | ~4.0s          |
| 1024x1024 | GFPGAN   | ~0.45s        | ~8.5s          |
| 1024x1024 | CodeFormer| ~0.75s       | ~14.0s         |

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t face-enhancement-ai .

# Run API server
docker run -p 8000:8000 --gpus all face-enhancement-ai

# Run CLI
docker run -v $(pwd)/data:/data face-enhancement-ai \
  face-enhance enhance /data/input.jpg --output /data/output.jpg
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/face_enhancement --cov-report=html

# Run specific tests
pytest tests/unit/test_face_detector.py
```

## 📚 API Documentation

Once the API server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🛠️ Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src tests
isort src tests

# Lint
flake8 src tests
mypy src

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## 📖 Model References

- **GFPGAN**: [Towards Real-World Blind Face Restoration with Generative Facial Prior](https://github.com/TencentARC/GFPGAN)
- **CodeFormer**: [Towards Robust Blind Face Restoration with Codebook Lookup Transformer](https://github.com/sczhou/CodeFormer)
- **Real-ESRGAN**: [Real-ESRGAN: Training Real-World Blind Super-Resolution](https://github.com/xinntao/Real-ESRGAN)
- **RetinaFace**: [RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild](https://github.com/biubug6/Pytorch_Retinaface)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TencentARC for GFPGAN
- NTU for CodeFormer
- Xintao Wang for Real-ESRGAN and BasicSR
- Face detection models from various researchers

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for better security camera footage analysis**
