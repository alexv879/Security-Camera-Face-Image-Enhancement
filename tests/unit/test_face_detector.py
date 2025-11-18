"""Unit tests for face detector."""

import numpy as np
import pytest

from face_enhancement.core.face_detector import FaceDetector


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def detector():
    """Create a face detector instance."""
    return FaceDetector(model="opencv", device="cpu")


def test_detector_initialization(detector):
    """Test detector initialization."""
    assert detector is not None
    assert detector.model_name == "opencv"
    assert detector.device == "cpu"


def test_detect_faces_returns_list(detector, sample_image):
    """Test that detect_faces returns a list."""
    result = detector.detect_faces(sample_image)
    assert isinstance(result, list)


def test_detect_faces_output_format(detector, sample_image):
    """Test the output format of face detection."""
    faces = detector.detect_faces(sample_image)

    for bbox, confidence, landmarks in faces:
        assert isinstance(bbox, np.ndarray)
        assert len(bbox) == 4
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


def test_get_face_crops(detector, sample_image):
    """Test face crop extraction."""
    crops = detector.get_face_crops(sample_image, padding=0.1)
    assert isinstance(crops, list)

    for crop, bbox in crops:
        assert isinstance(crop, np.ndarray)
        assert len(crop.shape) == 3
        assert isinstance(bbox, np.ndarray)
        assert len(bbox) == 4


def test_visualize_detections(detector, sample_image):
    """Test detection visualization."""
    result = detector.visualize_detections(sample_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape
