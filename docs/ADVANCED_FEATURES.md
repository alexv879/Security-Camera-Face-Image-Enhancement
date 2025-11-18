# Advanced Features Guide

This guide covers the advanced features of Face Enhancement AI.

## Table of Contents

1. [Model Ensemble](#model-ensemble)
2. [Advanced Preprocessing](#advanced-preprocessing)
3. [Quality Assessment](#quality-assessment)
4. [Performance Monitoring](#performance-monitoring)
5. [Caching](#caching)
6. [Model Optimization](#model-optimization)
7. [Benchmarking](#benchmarking)

---

## Model Ensemble

Combine multiple enhancement models for superior results.

### Usage

```python
from face_enhancement.models.face_enhancer import FaceEnhancer
from face_enhancement.models.ensemble import EnhancementEnsemble

# Create individual models
model1 = FaceEnhancer(model="gfpgan", upscale=2)
model2 = FaceEnhancer(model="codeformer", upscale=2)

# Create ensemble
ensemble = EnhancementEnsemble(
    models=[model1, model2],
    fusion_method="weighted_average",  # or 'quality_select', 'pixel_max'
    weights=[0.6, 0.4]  # Optional custom weights
)

# Use ensemble
enhanced, metadata = ensemble.enhance(image)
```

### Fusion Methods

- **weighted_average**: Blend models using specified weights
- **quality_select**: Choose best result based on quality metrics
- **pixel_max**: Take maximum value per pixel across all models

---

## Advanced Preprocessing

### HDR Tone Mapping

Apply HDR tone mapping for better dynamic range.

```python
from face_enhancement.preprocessing.advanced_preprocessor import AdvancedPreprocessor

preprocessor = AdvancedPreprocessor()

# Reinhard tone mapping
result = preprocessor.apply_hdr_tone_mapping(image, method="reinhard")

# Drago tone mapping (better for high dynamic range)
result = preprocessor.apply_hdr_tone_mapping(image, method="drago")

# Mantiuk tone mapping (preserves contrast)
result = preprocessor.apply_hdr_tone_mapping(image, method="mantiuk")
```

### Advanced Color Correction

```python
# Gray world assumption
result = preprocessor.apply_advanced_color_correction(image, method="gray_world")

# Max RGB correction
result = preprocessor.apply_advanced_color_correction(image, method="max_rgb")

# Multi-scale Retinex (illumination invariant)
result = preprocessor.apply_advanced_color_correction(image, method="retinex")
```

### Fog/Haze Removal

```python
# Remove fog using dark channel prior
result = preprocessor.apply_fog_removal(image, omega=0.95)
```

### Local Adaptive Enhancement

```python
# Apply local adaptive enhancement
result = preprocessor.apply_local_adaptive_enhancement(image, window_size=64)
```

---

## Quality Assessment

Evaluate image quality with multiple metrics.

```python
from face_enhancement.utils.quality_metrics import QualityAssessment

# Calculate all metrics for single image
metrics = QualityAssessment.calculate_all_metrics(image)
# Returns: {sharpness, brightness, contrast, colorfulness, entropy, niqe}

# Compare original and enhanced
comparison = QualityAssessment.compare_images(original, enhanced)
# Returns: {original: {...}, enhanced: {...}, improvement: {...}}

# Individual metrics
sharpness = QualityAssessment.calculate_sharpness(image)
psnr = QualityAssessment.calculate_psnr(reference, enhanced)
ssim = QualityAssessment.calculate_ssim(reference, enhanced)
```

### Available Metrics

- **PSNR**: Peak Signal-to-Noise Ratio (requires reference)
- **SSIM**: Structural Similarity Index (requires reference)
- **Sharpness**: Laplacian variance
- **Brightness**: Average luminance
- **Contrast**: RMS contrast
- **Colorfulness**: Colorfulness metric
- **Entropy**: Information content
- **NIQE**: Natural Image Quality Evaluator (no reference needed)

---

## Performance Monitoring

### Prometheus Metrics

The API server exposes Prometheus metrics at `/metrics`.

```python
from face_enhancement.utils.metrics import (
    track_request,
    record_faces_detected,
    record_image_size,
)

# Track request (decorator)
@track_request(model="gfpgan")
def process_image(image):
    # ... processing ...
    pass

# Record metrics manually
record_faces_detected(num_faces)
record_image_size(width, height)
```

### Available Metrics

- `face_enhancement_requests_total` - Total requests
- `face_enhancement_duration_seconds` - Processing time
- `face_enhancement_faces_detected` - Number of faces detected
- `face_enhancement_image_size_pixels` - Image size
- `face_enhancement_active_requests` - Active requests

### Grafana Dashboard

Import the provided dashboard configuration:

```bash
# In your Grafana instance
# Import dashboard from: config/grafana_dashboard.json
```

---

## Caching

Improve performance with intelligent caching.

### File-Based Cache

```python
from face_enhancement.utils.cache import Cache, cached

# Initialize cache
cache = Cache(backend="file", cache_dir=Path(".cache"))

# Use decorator
@cached(cache, ttl=3600)  # 1 hour TTL
def expensive_function(image):
    # ... expensive processing ...
    return result

# Manual cache operations
cache.set("key", value, ttl=3600)
result = cache.get("key")
cache.delete("key")
cache.clear()
```

### Redis Cache

```python
# Initialize Redis cache
cache = Cache(
    backend="redis",
    redis_host="localhost",
    redis_port=6379,
    redis_db=0,
)

# Use same API as file cache
@cached(cache, ttl=3600)
def process_image(image):
    # ... processing ...
    pass
```

---

## Model Optimization

### ONNX Export

Export models to ONNX for optimized deployment.

```python
from face_enhancement.models.optimization import ModelOptimizer

optimizer = ModelOptimizer()

# Export to ONNX
success = optimizer.export_to_onnx(
    model,
    output_path=Path("model.onnx"),
    input_shape=(1, 3, 512, 512),
    opset_version=14,
)
```

### ONNX Inference

```python
from face_enhancement.models.optimization import ONNXInference

# Load ONNX model
onnx_model = ONNXInference(Path("model.onnx"))

# Run inference
output = onnx_model.infer(input_array)

# Benchmark
results = onnx_model.benchmark(input_shape=(1, 3, 512, 512))
```

### Model Quantization

```python
# Dynamic quantization (easiest, good for CPU)
quantized_model = optimizer.quantize_model(
    model,
    output_path=Path("model_quantized.pt"),
    quantization_type="dynamic",
)

# Static quantization (best compression, needs calibration)
quantized_model = optimizer.quantize_model(
    model,
    quantization_type="static",
)
```

### Command-Line Export

```bash
# Export model to ONNX
python scripts/export_onnx.py \
    --model-path models/gfpgan.pth \
    --output models/gfpgan.onnx \
    --input-shape 1 3 512 512 \
    --benchmark
```

---

## Benchmarking

### Comprehensive Benchmarks

```bash
# Run full benchmark suite
python scripts/benchmark.py \
    --device cuda \
    --model gfpgan \
    --runs 100 \
    --output benchmark_results/
```

### Programmatic Benchmarking

```python
from face_enhancement.utils.benchmark import PerformanceBenchmark

benchmark = PerformanceBenchmark()

# Benchmark function
result = benchmark.benchmark_function(
    func=my_function,
    args=(arg1, arg2),
    kwargs={"param": value},
    name="my_benchmark",
    num_runs=100,
    warmup_runs=10,
)

# Benchmark pipeline with variations
variations = {
    "upscale_2x": {"enhancement.upscale_factor": 2},
    "upscale_4x": {"enhancement.upscale_factor": 4},
}

results = benchmark.benchmark_pipeline(
    pipeline,
    test_images,
    variations=variations,
)

# Generate report
report = benchmark.generate_report(Path("report.txt"))
benchmark.save_results_json(Path("results.json"))
```

### Quality Comparison Script

```bash
# Compare image quality before/after
python scripts/quality_comparison.py input.jpg \
    --output enhanced.jpg \
    --model gfpgan \
    --upscale 2
```

---

## API Advanced Features

### Quality Assessment Endpoint

```bash
# Assess image quality
curl -X POST "http://localhost:8000/quality" \
    -F "file=@image.jpg"
```

### Image Comparison Endpoint

```bash
# Compare two images
curl -X POST "http://localhost:8000/compare" \
    -F "original=@original.jpg" \
    -F "enhanced=@enhanced.jpg"
```

### Metrics Endpoint

```bash
# Get Prometheus metrics
curl http://localhost:8000/metrics
```

---

## Best Practices

### Performance

1. **Use GPU** when available for 10-20x speedup
2. **Enable caching** for repeated operations
3. **Batch processing** for multiple images
4. **ONNX export** for production deployment
5. **Model quantization** for edge devices

### Quality

1. **Ensemble models** for best quality
2. **Tune preprocessing** for your specific use case
3. **Monitor quality metrics** continuously
4. **A/B test** different configurations
5. **Use quality assessment** to validate improvements

### Production

1. **Monitor with Prometheus** and Grafana
2. **Use Docker** for consistent deployment
3. **Enable caching** with Redis
4. **Load balance** API servers
5. **Set up alerts** for failures

---

## Troubleshooting

### Out of Memory

- Reduce upscale factor
- Process smaller batches
- Use CPU for large images
- Enable model quantization

### Slow Performance

- Enable GPU if available
- Use ONNX export
- Enable caching
- Reduce preprocessing steps
- Use smaller models

### Quality Issues

- Try ensemble mode
- Adjust preprocessing parameters
- Use quality metrics to guide tuning
- Test different models
- Adjust fidelity weight (CodeFormer)

---

## Next Steps

- Explore [API Documentation](API.md)
- Read [Installation Guide](INSTALLATION.md)
- Check [Contributing Guide](../CONTRIBUTING.md)
- See [Examples](../examples/example_usage.py)
