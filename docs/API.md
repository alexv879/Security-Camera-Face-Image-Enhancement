# API Documentation

## REST API Reference

The Face Enhancement AI REST API provides endpoints for face detection and enhancement.

### Base URL

```
http://localhost:8000
```

### Authentication

Currently no authentication required. Add authentication in production deployments.

---

## Endpoints

### GET /

Root endpoint with service information.

**Response:**
```json
{
  "service": "Face Enhancement API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "enhance": "/enhance",
    "enhance_batch": "/enhance/batch",
    "health": "/health",
    "info": "/info"
  }
}
```

---

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "pipeline_initialized": true,
  "device": "cuda"
}
```

---

### GET /info

Get pipeline configuration and model information.

**Response:**
```json
{
  "preprocessor": {
    "denoise": true,
    "auto_contrast": true,
    "enhance_brightness": true,
    "gamma": 1.0,
    "sharpen": false
  },
  "detector": {
    "model": "retinaface",
    "device": "cuda",
    "confidence_threshold": 0.5
  },
  "enhancer": {
    "model": "gfpgan",
    "upscale": 2,
    "device": "cuda",
    "bg_upsampler": "realesrgan",
    "initialized": true
  }
}
```

---

### POST /enhance

Enhance a single image.

**Parameters:**
- `file` (required): Image file (multipart/form-data)
- `model` (optional): Enhancement model (`gfpgan` or `codeformer`), default: `gfpgan`
- `upscale` (optional): Upscaling factor (1-4), default: `2`
- `preprocess` (optional): Apply preprocessing, default: `true`

**Example Request:**
```bash
curl -X POST "http://localhost:8000/enhance" \
  -F "file=@input.jpg" \
  -F "model=gfpgan" \
  -F "upscale=2" \
  --output enhanced.jpg
```

**Response:**
- Content-Type: `image/png`
- Headers:
  - `X-Num-Faces`: Number of faces detected
  - `X-Processing-Time`: Processing time in seconds
  - `X-Original-Shape`: Original image dimensions
  - `X-Output-Shape`: Output image dimensions

---

### POST /enhance/metadata

Enhance image and return metadata without the image.

**Parameters:**
Same as `/enhance`

**Example Request:**
```bash
curl -X POST "http://localhost:8000/enhance/metadata" \
  -F "file=@input.jpg" \
  -F "model=gfpgan"
```

**Response:**
```json
{
  "success": true,
  "num_faces": 2,
  "original_shape": [480, 640, 3],
  "output_shape": [960, 1280, 3],
  "processing_time": 1.234,
  "message": "Enhancement successful"
}
```

---

### POST /detect

Detect faces in image without enhancement.

**Parameters:**
- `file` (required): Image file

**Example Request:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@input.jpg"
```

**Response:**
```json
{
  "num_faces": 2,
  "image_shape": [480, 640, 3],
  "faces": [
    {
      "bbox": [100, 120, 250, 300],
      "confidence": 0.98,
      "landmarks": [[150, 170], [200, 175], [175, 210], [160, 250], [190, 255]]
    }
  ]
}
```

---

## Python Client Example

```python
import requests

# Enhance image
with open('input.jpg', 'rb') as f:
    files = {'file': f}
    data = {'model': 'gfpgan', 'upscale': 2}
    response = requests.post('http://localhost:8000/enhance', files=files, data=data)

with open('output.jpg', 'wb') as f:
    f.write(response.content)

# Get metadata
num_faces = response.headers.get('X-Num-Faces')
processing_time = response.headers.get('X-Processing-Time')
print(f"Detected {num_faces} faces in {processing_time}s")
```

---

## JavaScript Client Example

```javascript
async function enhanceImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', 'gfpgan');
  formData.append('upscale', '2');

  const response = await fetch('http://localhost:8000/enhance', {
    method: 'POST',
    body: formData
  });

  const blob = await response.blob();
  const numFaces = response.headers.get('X-Num-Faces');

  console.log(`Detected ${numFaces} faces`);
  return blob;
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid image file"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Enhancement failed: [error message]"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Pipeline not initialized"
}
```

---

## Rate Limiting

No rate limiting currently implemented. Consider adding rate limiting in production:
- Use FastAPI middleware (slowapi)
- Configure per-endpoint limits
- Return 429 Too Many Requests when exceeded

---

## CORS

For web applications, configure CORS in the FastAPI app:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Deployment

### Docker

```bash
docker run -p 8000:8000 --gpus all face-enhancement-ai
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-enhancement-api
spec:
  replicas: 3
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
```

---

## Performance Tips

1. **Use GPU** - Enable CUDA for 10-20x faster processing
2. **Batch requests** - Process multiple images together
3. **Cache models** - Load models once, reuse for requests
4. **Async processing** - Use background tasks for long operations
5. **Image compression** - Compress images before upload

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/alexv879/Security-Camera-Face-Image-Enhancement/issues
- Documentation: https://github.com/alexv879/Security-Camera-Face-Image-Enhancement/docs
