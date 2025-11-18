## Real-Time Streaming Enhancement Guide

This guide covers real-time video stream enhancement capabilities.

## Overview

The real-time streaming module provides:
- Live video enhancement with minimal latency
- Adaptive quality based on processing speed
- Temporal smoothing for consistent results
- Frame buffering and dropping
- Multi-threaded processing

## Quick Start

### Command Line

```bash
# Enhance webcam feed (camera 0)
python scripts/stream_enhance.py --source 0 --fps 15

# Enhance video file with output
python scripts/stream_enhance.py \
    --source surveillance.mp4 \
    --output enhanced_output.mp4 \
    --fps 20

# Process RTSP stream
python scripts/stream_enhance.py \
    --source "rtsp://camera.local/stream" \
    --fps 10
```

### Python API

```python
from face_enhancement.streaming import VideoStreamEnhancer

# Create enhancer
enhancer = VideoStreamEnhancer(
    video_source="0",  # Webcam
    output_path="output.mp4",  # Optional
    target_fps=15,
    adaptive_quality=True,
)

# Run with display
enhancer.run(display=True, max_frames=1000)
```

## Advanced Usage

### Custom Processing Callback

```python
def my_callback(original, enhanced, frame_id):
    """Process each enhanced frame."""
    # Save specific frames
    if frame_id % 30 == 0:
        cv2.imwrite(f"frame_{frame_id}.jpg", enhanced)

    # Analyze quality
    from face_enhancement.utils.quality_metrics import QualityAssessment
    metrics = QualityAssessment.calculate_all_metrics(enhanced)
    print(f"Frame {frame_id}: sharpness={metrics['sharpness']:.2f}")

enhancer.run(callback=my_callback)
```

### Low-Level API

```python
from face_enhancement.streaming import RealtimeEnhancer
from face_enhancement.config import AppConfig

# Create enhancer
config = AppConfig()
enhancer = RealtimeEnhancer(
    config=config,
    max_buffer_size=30,
    target_fps=15,
    enable_temporal_smoothing=True,
)

# Start processing thread
enhancer.start()

# Process frames
for frame_id, frame in enumerate(video_frames):
    # Add to queue
    success = enhancer.process_frame(frame, frame_id)

    # Get result
    result = enhancer.get_enhanced_frame(timeout=0.1)
    if result:
        enhanced, fid = result
        display_frame(enhanced)

# Stop
enhancer.stop()

# Get statistics
stats = enhancer.get_stats()
print(f"Processed: {stats['frames_processed']}")
print(f"Dropped: {stats['frames_dropped']}")
print(f"FPS: {stats['current_fps']:.1f}")
print(f"Latency: {stats['avg_latency']:.3f}s")
```

## Features

### Temporal Smoothing

Smooths enhancement between consecutive frames to reduce flicker:

```python
enhancer = RealtimeEnhancer(
    enable_temporal_smoothing=True,  # Enable smoothing
)
```

The smoothing uses exponential moving average:
- α = 0.7 for current frame
- (1-α) = 0.3 for previous frame

### Adaptive Quality

Automatically adjusts processing based on latency:

```python
enhancer = VideoStreamEnhancer(
    adaptive_quality=True,  # Enable adaptation
)
```

Adaptation strategies:
- High latency → Skip preprocessing
- Very high latency → Reduce upscale factor
- Extreme latency → Drop frames

### Frame Buffering

Configurable input/output buffers:

```python
enhancer = RealtimeEnhancer(
    max_buffer_size=30,  # Max 30 frames buffered
)
```

When buffer is full:
- Input: Drops new frames
- Output: Drops oldest frames

## Performance Optimization

### 1. Reduce Upscale Factor

```python
config = AppConfig()
config.enhancement.upscale_factor = 2  # Instead of 4
```

### 2. Disable Preprocessing

```python
# In pipeline
enhanced, _ = pipeline.process_image(
    frame,
    preprocess=False,  # Skip preprocessing
)
```

### 3. Use ONNX Models

```python
# Export model to ONNX first
python scripts/export_onnx.py \
    --model-path models/gfpgan.pth \
    --output models/gfpgan.onnx

# Then use ONNX inference (2-3x faster)
from face_enhancement.models.optimization import ONNXInference
onnx_model = ONNXInference("models/gfpgan.onnx")
```

### 4. Adjust Target FPS

```python
enhancer = VideoStreamEnhancer(
    target_fps=10,  # Lower FPS = more processing time per frame
)
```

## Monitoring

### Real-Time Statistics

```python
import time

while processing:
    time.sleep(5)  # Every 5 seconds

    stats = enhancer.get_stats()
    print(f"""
    Frames Processed: {stats['frames_processed']}
    Frames Dropped: {stats['frames_dropped']}
    Current FPS: {stats['current_fps']:.1f}
    Average Latency: {stats['avg_latency']:.3f}s
    """)
```

### Prometheus Metrics

The streaming module integrates with Prometheus metrics:

```python
from face_enhancement.utils.metrics import (
    record_faces_detected,
    record_image_size,
)

# Record metrics for each frame
for frame in frames:
    enhanced, metadata = process(frame)
    record_faces_detected(metadata['num_faces'])
    record_image_size(frame.shape[1], frame.shape[0])
```

## Use Cases

### 1. Security Camera Monitoring

```python
# Monitor multiple RTSP streams
streams = [
    "rtsp://camera1.local/stream",
    "rtsp://camera2.local/stream",
    "rtsp://camera3.local/stream",
]

for stream_url in streams:
    enhancer = VideoStreamEnhancer(stream_url, target_fps=10)
    enhancer.run(display=False, callback=save_alerts)
```

### 2. Video Conferencing Enhancement

```python
# Enhance webcam for better video calls
enhancer = VideoStreamEnhancer(
    video_source="0",
    target_fps=30,  # Higher FPS for video calls
    adaptive_quality=True,
)

# Virtual camera output (requires pyvirtualcam)
import pyvirtualcam

with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
    def send_to_virtual_cam(original, enhanced, frame_id):
        cam.send(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

    enhancer.run(callback=send_to_virtual_cam)
```

### 3. Batch Video Processing

```python
import glob

video_files = glob.glob("videos/*.mp4")

for video_file in video_files:
    output_file = f"enhanced_{Path(video_file).name}"

    enhancer = VideoStreamEnhancer(
        video_source=video_file,
        output_path=output_file,
        target_fps=15,
    )

    enhancer.run(display=False)
    print(f"Processed: {video_file} → {output_file}")
```

## Troubleshooting

### High Latency

**Problem**: Latency > 1 second

**Solutions**:
1. Reduce target FPS
2. Lower upscale factor
3. Disable preprocessing
4. Use GPU if available
5. Use ONNX models

### Frame Drops

**Problem**: Many dropped frames

**Solutions**:
1. Increase buffer size
2. Reduce processing complexity
3. Lower input resolution
4. Use faster hardware

### Poor Quality

**Problem**: Output quality degraded

**Solutions**:
1. Increase upscale factor
2. Enable preprocessing
3. Disable adaptive quality
4. Use higher model fidelity

### Memory Issues

**Problem**: Out of memory

**Solutions**:
1. Reduce buffer size
2. Lower resolution
3. Reduce upscale factor
4. Process fewer frames

## Best Practices

1. **Start with low FPS**: Begin with 10-15 FPS and increase if latency allows
2. **Monitor statistics**: Regularly check frame drops and latency
3. **Use GPU**: Always use GPU for real-time processing
4. **Enable temporal smoothing**: Reduces flicker in video
5. **Tune buffer size**: Balance latency vs robustness
6. **Test thoroughly**: Different videos have different characteristics

## Performance Targets

| Hardware | Resolution | FPS | Latency |
|----------|-----------|-----|---------|
| RTX 3090 | 720p | 20-25 | <100ms |
| RTX 3080 | 720p | 15-20 | <150ms |
| RTX 2080 | 720p | 10-15 | <200ms |
| CPU (i7) | 720p | 2-3 | >1s |

## Next Steps

- Explore [Advanced Features](ADVANCED_FEATURES.md)
- Check [API Documentation](API.md)
- See [Progressive Enhancement](../README.md#progressive-enhancement)
