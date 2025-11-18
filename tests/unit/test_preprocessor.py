"""Unit tests for image preprocessor."""

import numpy as np
import pytest

from face_enhancement.preprocessing.image_preprocessor import ImagePreprocessor


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance."""
    return ImagePreprocessor(
        denoise=True,
        auto_contrast=True,
        enhance_brightness=True,
        gamma=1.0,
        sharpen=False,
    )


def test_preprocessor_initialization(preprocessor):
    """Test preprocessor initialization."""
    assert preprocessor is not None
    assert preprocessor.denoise is True
    assert preprocessor.auto_contrast is True


def test_preprocess_returns_ndarray(preprocessor, sample_image):
    """Test that preprocess returns numpy array."""
    result = preprocessor.preprocess(sample_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape


def test_apply_denoising(preprocessor, sample_image):
    """Test denoising."""
    result = preprocessor.apply_denoising(sample_image, method="bilateral")
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape


def test_enhance_low_light(preprocessor, sample_image):
    """Test low-light enhancement."""
    result = preprocessor.enhance_low_light(sample_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape


def test_apply_auto_contrast(preprocessor, sample_image):
    """Test auto contrast."""
    result = preprocessor.apply_auto_contrast(sample_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape


def test_apply_gamma_correction(preprocessor, sample_image):
    """Test gamma correction."""
    result = preprocessor.apply_gamma_correction(sample_image, gamma=1.2)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape


def test_apply_sharpening(preprocessor, sample_image):
    """Test sharpening."""
    result = preprocessor.apply_sharpening(sample_image, method="unsharp")
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape


def test_auto_white_balance(preprocessor, sample_image):
    """Test auto white balance."""
    result = preprocessor.auto_white_balance(sample_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape
