"""Unit tests for configuration."""

import tempfile
from pathlib import Path

import pytest

from face_enhancement.config import AppConfig, FaceDetectionConfig, EnhancementConfig


def test_default_config():
    """Test default configuration."""
    config = AppConfig()
    assert config is not None
    assert config.device == "cuda"
    assert config.log_level == "INFO"


def test_face_detection_config():
    """Test face detection configuration."""
    config = FaceDetectionConfig(
        model="retinaface",
        confidence_threshold=0.7,
        nms_threshold=0.4,
    )
    assert config.model == "retinaface"
    assert config.confidence_threshold == 0.7
    assert config.nms_threshold == 0.4


def test_enhancement_config():
    """Test enhancement configuration."""
    config = EnhancementConfig(
        model="gfpgan",
        upscale_factor=2,
        face_upsample=True,
    )
    assert config.model == "gfpgan"
    assert config.upscale_factor == 2
    assert config.face_upsample is True


def test_config_to_yaml():
    """Test configuration export to YAML."""
    config = AppConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "config.yaml"
        config.to_yaml(yaml_path)
        assert yaml_path.exists()


def test_config_from_yaml():
    """Test configuration import from YAML."""
    config = AppConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "config.yaml"
        config.to_yaml(yaml_path)

        loaded_config = AppConfig.from_yaml(yaml_path)
        assert loaded_config.device == config.device
        assert loaded_config.log_level == config.log_level


def test_get_device():
    """Test device selection."""
    config = AppConfig(device="cuda")
    device = config.get_device()
    assert device in ["cuda", "cpu"]
