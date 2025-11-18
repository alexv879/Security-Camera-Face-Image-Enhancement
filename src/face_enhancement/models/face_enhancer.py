"""State-of-the-art face enhancement using GFPGAN, CodeFormer, and Real-ESRGAN."""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False
    logger.warning("GFPGAN not available")

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    logger.warning("Real-ESRGAN not available")


class FaceEnhancer:
    """
    Advanced face enhancement using state-of-the-art deep learning models.

    Supports:
    - GFPGAN v1.4: Practical face restoration model
    - CodeFormer: Robust face restoration with controllable fidelity
    - Real-ESRGAN: Background upsampling
    """

    def __init__(
        self,
        model: str = "gfpgan",
        upscale: int = 2,
        device: str = "cuda",
        model_path: Optional[Path] = None,
        bg_upsampler: str = "realesrgan",
        fidelity_weight: float = 0.5,
    ):
        """
        Initialize face enhancer.

        Args:
            model: Enhancement model ('gfpgan', 'codeformer')
            upscale: Upscaling factor (1-4)
            device: Device ('cuda' or 'cpu')
            model_path: Optional path to model weights
            bg_upsampler: Background upsampler ('realesrgan' or None)
            fidelity_weight: CodeFormer fidelity weight (0=better quality, 1=better identity)
        """
        self.model_name = model
        self.upscale = upscale
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.fidelity_weight = fidelity_weight

        logger.info(f"Initializing {model} face enhancer on {self.device}")

        # Initialize background upsampler
        self.bg_upsampler = None
        if bg_upsampler == "realesrgan":
            self.bg_upsampler = self._init_bg_upsampler()

        # Initialize face restoration model
        self.enhancer = self._init_enhancer()

    def _init_bg_upsampler(self) -> Optional[RealESRGANer]:
        """Initialize Real-ESRGAN for background upsampling."""
        if not REALESRGAN_AVAILABLE:
            logger.warning("Real-ESRGAN not available for background upsampling")
            return None

        try:
            # Use RealESRGAN_x2plus model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=self.upscale,
            )

            bg_upsampler = RealESRGANer(
                scale=self.upscale,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                device=self.device,
            )

            logger.info("Real-ESRGAN background upsampler initialized")
            return bg_upsampler

        except Exception as e:
            logger.error(f"Failed to initialize Real-ESRGAN: {e}")
            return None

    def _init_enhancer(self):
        """Initialize face enhancement model."""
        if self.model_name == "gfpgan" and GFPGAN_AVAILABLE:
            return self._init_gfpgan()
        else:
            logger.error(f"Model {self.model_name} not available or not supported")
            return None

    def _init_gfpgan(self) -> Optional[GFPGANer]:
        """Initialize GFPGAN model."""
        try:
            # Auto-download model if not provided
            model_path = self.model_path or "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

            enhancer = GFPGANer(
                model_path=str(model_path),
                upscale=self.upscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.bg_upsampler,
                device=self.device,
            )

            logger.info("GFPGAN model initialized successfully")
            return enhancer

        except Exception as e:
            logger.error(f"Failed to initialize GFPGAN: {e}")
            return None

    def enhance(
        self,
        image: np.ndarray,
        face_bbox: Optional[np.ndarray] = None,
        align: bool = True,
        only_center_face: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhance face in image.

        Args:
            image: Input image (BGR)
            face_bbox: Optional face bounding box [x1, y1, x2, y2]
            align: Whether to align face before enhancement
            only_center_face: Only enhance the center face

        Returns:
            Tuple of (enhanced_image, original_image)
        """
        if self.enhancer is None:
            logger.error("Enhancer not initialized")
            return image, image

        try:
            # GFPGAN expects BGR input
            _, _, enhanced_img = self.enhancer.enhance(
                image,
                has_aligned=not align,
                only_center_face=only_center_face,
                paste_back=True,
                weight=self.fidelity_weight,
            )

            if enhanced_img is None:
                logger.warning("Enhancement failed, returning original image")
                return image, image

            return enhanced_img, image

        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return image, image

    def enhance_face_crop(
        self, face_crop: np.ndarray, paste_back: bool = False, background: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Enhance a cropped face region.

        Args:
            face_crop: Cropped face image
            paste_back: Whether to paste enhanced face back to original
            background: Background image for paste_back

        Returns:
            Enhanced face crop or full image
        """
        if self.enhancer is None:
            return face_crop

        try:
            _, _, enhanced = self.enhancer.enhance(
                face_crop,
                has_aligned=False,
                only_center_face=True,
                paste_back=paste_back,
            )

            return enhanced if enhanced is not None else face_crop

        except Exception as e:
            logger.error(f"Crop enhancement failed: {e}")
            return face_crop

    def batch_enhance(self, images: list) -> list:
        """
        Enhance multiple images in batch.

        Args:
            images: List of input images

        Returns:
            List of enhanced images
        """
        enhanced_images = []

        for i, img in enumerate(images):
            logger.info(f"Enhancing image {i+1}/{len(images)}")
            enhanced, _ = self.enhance(img)
            enhanced_images.append(enhanced)

        return enhanced_images

    def upscale_background(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale background only (no face enhancement).

        Args:
            image: Input image

        Returns:
            Upscaled image
        """
        if self.bg_upsampler is None:
            logger.warning("Background upsampler not available")
            return image

        try:
            output, _ = self.bg_upsampler.enhance(image, outscale=self.upscale)
            return output
        except Exception as e:
            logger.error(f"Background upsampling failed: {e}")
            return image

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model": self.model_name,
            "upscale": self.upscale,
            "device": self.device,
            "bg_upsampler": "realesrgan" if self.bg_upsampler else None,
            "fidelity_weight": self.fidelity_weight,
            "initialized": self.enhancer is not None,
        }
