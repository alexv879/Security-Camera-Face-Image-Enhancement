# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-18

### Added
- Initial release of Face Enhancement AI
- State-of-the-art face detection using RetinaFace
- Face enhancement with GFPGAN and CodeFormer models
- Real-ESRGAN background super-resolution
- Advanced preprocessing pipeline:
  - Low-light enhancement (CLAHE)
  - Denoising (bilateral filter, NLM)
  - Auto contrast and brightness adjustment
  - Gamma correction
  - Sharpening
- Command-line interface with rich output
- FastAPI REST API server
- Batch processing support
- Video frame extraction and enhancement
- Comprehensive configuration system (YAML, environment variables)
- Docker and Docker Compose support
- Unit and integration tests
- Pre-commit hooks for code quality
- CI/CD with GitHub Actions
- Comprehensive documentation:
  - README with quick start guide
  - API documentation
  - Contributing guide
  - Example usage scripts

### Features
- GPU acceleration (CUDA support)
- CPU fallback for systems without GPU
- Configurable enhancement models and parameters
- Face detection with landmark points
- Before/after comparison images
- Detection visualization
- Progress tracking for batch operations
- Extensive logging with loguru
- Type hints throughout codebase

### Models Supported
- **Face Detection**: RetinaFace, OpenCV Haar Cascade
- **Face Enhancement**: GFPGAN v1.4, CodeFormer
- **Background Upsampling**: Real-ESRGAN x2plus

### API Endpoints
- `GET /` - Service information
- `GET /health` - Health check
- `GET /info` - Pipeline configuration
- `POST /enhance` - Enhance single image
- `POST /enhance/metadata` - Get enhancement metadata
- `POST /detect` - Detect faces only

### CLI Commands
- `face-enhance enhance` - Enhance single image
- `face-enhance batch` - Batch process images
- `face-enhance video` - Process video frames
- `face-enhance info` - Show pipeline info

### Dependencies
- PyTorch >= 2.0.0
- OpenCV >= 4.8.0
- GFPGAN >= 1.3.8
- BasicSR >= 1.4.2
- Real-ESRGAN >= 0.3.0
- FastAPI >= 0.103.0
- Rich >= 13.5.0
- And more (see requirements.txt)

## [Unreleased]

### Planned
- CodeFormer model integration
- Additional face detection models (YOLOv5Face, SCRFD)
- Video output (reassemble enhanced frames)
- Web UI dashboard
- Model quantization for faster inference
- TensorRT optimization
- ONNX export for cross-platform deployment
- Batch API endpoint
- Asynchronous processing queue
- Results caching
- Metrics and monitoring dashboard
- User authentication for API
- Rate limiting
- S3/cloud storage integration

[1.0.0]: https://github.com/alexv879/Security-Camera-Face-Image-Enhancement/releases/tag/v1.0.0
