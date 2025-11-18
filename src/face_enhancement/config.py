"""Configuration management for face enhancement system."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class FaceDetectionConfig(BaseModel):
    """Configuration for face detection."""

    model: str = Field(default="retinaface", description="Face detection model")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    device: str = Field(default="cuda", description="Device: cuda or cpu")
    max_faces: int = Field(default=10, gt=0)


class EnhancementConfig(BaseModel):
    """Configuration for face enhancement."""

    model: str = Field(default="gfpgan", description="Enhancement model: gfpgan, codeformer")
    upscale_factor: int = Field(default=2, ge=1, le=4)
    face_upsample: bool = Field(default=True)
    bg_upsampler: Optional[str] = Field(default="realesrgan", description="Background upsampler")
    bg_tile: int = Field(default=400, gt=0)
    weight: float = Field(default=0.5, ge=0.0, le=1.0, description="CodeFormer fidelity weight")


class PreprocessingConfig(BaseModel):
    """Configuration for image preprocessing."""

    denoise: bool = Field(default=True)
    auto_contrast: bool = Field(default=True)
    enhance_brightness: bool = Field(default=True)
    gamma_correction: float = Field(default=1.0, ge=0.1, le=3.0)
    sharpen: bool = Field(default=False)


class ProcessingConfig(BaseModel):
    """Configuration for processing pipeline."""

    batch_size: int = Field(default=1, gt=0)
    num_workers: int = Field(default=4, ge=0)
    output_format: str = Field(default="png", pattern="^(png|jpg|jpeg)$")
    output_quality: int = Field(default=95, ge=1, le=100)
    save_comparison: bool = Field(default=True)
    save_detections: bool = Field(default=True)


class ModelPaths(BaseModel):
    """Paths to model weights."""

    detection_model: Optional[Path] = None
    gfpgan_model: Optional[Path] = None
    codeformer_model: Optional[Path] = None
    realesrgan_model: Optional[Path] = None
    parsing_model: Optional[Path] = None


class AppConfig(BaseSettings):
    """Main application configuration."""

    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    models_dir: Path = Field(default_factory=lambda: Path("models/weights"))
    output_dir: Path = Field(default_factory=lambda: Path("data/output"))
    log_dir: Path = Field(default_factory=lambda: Path("logs"))

    # Component configs
    face_detection: FaceDetectionConfig = Field(default_factory=FaceDetectionConfig)
    enhancement: EnhancementConfig = Field(default_factory=EnhancementConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    model_paths: ModelPaths = Field(default_factory=ModelPaths)

    # System settings
    device: str = Field(default="cuda")
    log_level: str = Field(default="INFO")
    enable_metrics: bool = Field(default=True)

    class Config:
        env_prefix = "FACE_ENHANCE_"
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def get_device(self) -> str:
        """Get the appropriate device for processing."""
        import torch

        if self.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"


# Default configuration instance
default_config = AppConfig()


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from file or use default.

    Args:
        config_path: Optional path to configuration YAML file

    Returns:
        AppConfig instance
    """
    if config_path and config_path.exists():
        return AppConfig.from_yaml(config_path)
    return AppConfig()
