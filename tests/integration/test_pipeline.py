"""Integration tests for enhancement pipeline."""

import numpy as np
import pytest

from face_enhancement.config import AppConfig
from face_enhancement.core.pipeline import EnhancementPipeline


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def pipeline():
    """Create a pipeline instance."""
    config = AppConfig()
    config.device = "cpu"  # Use CPU for testing
    config.face_detection.model = "opencv"  # Use fast detector
    return EnhancementPipeline(config)


def test_pipeline_initialization(pipeline):
    """Test pipeline initialization."""
    assert pipeline is not None
    assert pipeline.preprocessor is not None
    assert pipeline.detector is not None
    assert pipeline.enhancer is not None


def test_process_image(pipeline, sample_image):
    """Test image processing."""
    enhanced, metadata = pipeline.process_image(
        sample_image,
        preprocess=True,
        detect_faces=True,
        enhance=False,  # Skip enhancement to speed up test
    )

    assert isinstance(enhanced, np.ndarray)
    assert isinstance(metadata, dict)
    assert "num_faces" in metadata
    assert "original_shape" in metadata


def test_pipeline_info(pipeline):
    """Test pipeline info retrieval."""
    info = pipeline.get_pipeline_info()
    assert isinstance(info, dict)
    assert "preprocessor" in info
    assert "detector" in info
    assert "enhancer" in info
