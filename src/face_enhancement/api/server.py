"""FastAPI REST API server for face enhancement."""

import io
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from face_enhancement.config import AppConfig
from face_enhancement.core.pipeline import EnhancementPipeline
from face_enhancement.utils.logger import setup_logger


# Initialize FastAPI app
app = FastAPI(
    title="Face Enhancement API",
    description="State-of-the-art face image enhancement for security camera footage",
    version="1.0.0",
)

# Global pipeline instance
pipeline: Optional[EnhancementPipeline] = None
config: Optional[AppConfig] = None


class EnhanceRequest(BaseModel):
    """Request model for image enhancement."""

    model: str = Field(default="gfpgan", description="Enhancement model")
    upscale: int = Field(default=2, ge=1, le=4, description="Upscaling factor")
    preprocess: bool = Field(default=True, description="Apply preprocessing")
    denoise: bool = Field(default=True, description="Apply denoising")
    auto_contrast: bool = Field(default=True, description="Apply auto contrast")


class EnhanceResponse(BaseModel):
    """Response model for image enhancement."""

    success: bool
    num_faces: int
    original_shape: tuple
    output_shape: tuple
    processing_time: float
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline, config

    logger.info("Starting Face Enhancement API server")
    setup_logger(log_level="INFO")

    config = AppConfig()
    pipeline = EnhancementPipeline(config)

    logger.info("API server ready")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Face Enhancement API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "enhance": "/enhance",
            "enhance_batch": "/enhance/batch",
            "health": "/health",
            "info": "/info",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline is not None,
        "device": config.get_device() if config else "unknown",
    }


@app.get("/info")
async def get_info():
    """Get pipeline information."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return pipeline.get_pipeline_info()


@app.post("/enhance", response_model=EnhanceResponse)
async def enhance_image(
    file: UploadFile = File(...),
    model: str = "gfpgan",
    upscale: int = 2,
    preprocess: bool = True,
):
    """
    Enhance a single image.

    Args:
        file: Image file to enhance
        model: Enhancement model (gfpgan, codeformer)
        upscale: Upscaling factor (1-4)
        preprocess: Apply preprocessing

    Returns:
        Enhanced image
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        start_time = time.time()

        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Update config
        pipeline.config.enhancement.model = model
        pipeline.config.enhancement.upscale_factor = upscale

        # Process image
        enhanced, metadata = pipeline.process_image(
            image,
            preprocess=preprocess,
            detect_faces=True,
            enhance=True,
        )

        processing_time = time.time() - start_time

        # Encode result
        _, buffer = cv2.imencode(".png", enhanced)
        io_buf = io.BytesIO(buffer)

        return StreamingResponse(
            io_buf,
            media_type="image/png",
            headers={
                "X-Num-Faces": str(metadata["num_faces"]),
                "X-Processing-Time": str(processing_time),
                "X-Original-Shape": str(metadata["original_shape"]),
                "X-Output-Shape": str(metadata["output_shape"]),
            },
        )

    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enhance/metadata", response_model=EnhanceResponse)
async def enhance_image_with_metadata(
    file: UploadFile = File(...),
    model: str = "gfpgan",
    upscale: int = 2,
    preprocess: bool = True,
):
    """
    Enhance image and return metadata (no image).

    Args:
        file: Image file to enhance
        model: Enhancement model
        upscale: Upscaling factor
        preprocess: Apply preprocessing

    Returns:
        Enhancement metadata
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        start_time = time.time()

        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Update config
        pipeline.config.enhancement.model = model
        pipeline.config.enhancement.upscale_factor = upscale

        # Process image
        enhanced, metadata = pipeline.process_image(
            image,
            preprocess=preprocess,
            detect_faces=True,
            enhance=True,
        )

        processing_time = time.time() - start_time

        return EnhanceResponse(
            success=True,
            num_faces=metadata["num_faces"],
            original_shape=metadata["original_shape"],
            output_shape=metadata["output_shape"],
            processing_time=processing_time,
            message="Enhancement successful",
        )

    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        return EnhanceResponse(
            success=False,
            num_faces=0,
            original_shape=(0, 0, 0),
            output_shape=(0, 0, 0),
            processing_time=0.0,
            message=str(e),
        )


@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """
    Detect faces in image without enhancement.

    Args:
        file: Image file

    Returns:
        Face detection results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Detect faces
        faces = pipeline.detector.detect_faces(image)

        results = []
        for bbox, confidence, landmarks in faces:
            face_data = {
                "bbox": bbox.tolist(),
                "confidence": float(confidence),
                "landmarks": landmarks.tolist() if landmarks is not None else None,
            }
            results.append(face_data)

        return {
            "num_faces": len(faces),
            "faces": results,
            "image_shape": image.shape,
        }

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "face_enhancement.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
