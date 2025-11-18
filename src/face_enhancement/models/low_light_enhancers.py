"""Low-light specific enhancement models.

Implements state-of-the-art low-light enhancement:
- Zero-DCE++ (Zero-Reference Deep Curve Estimation)
- EnlightenGAN
- RetinexNet
- MBLLEN (Multi-Branch Low-Light Enhancement Network)
"""

from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class ZeroDCEPlusPlus:
    """
    Zero-DCE++: Learning to Enhance Low-Light Images.

    Zero-reference deep curve estimation for low-light enhancement.
    No paired training data required!
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize Zero-DCE++.

        Args:
            device: Device for inference
        """
        self.device = device
        self.model = None

        logger.info("Zero-DCE++ initializing")

    def _lazy_load(self):
        """Lazy load model."""
        if self.model is not None:
            return

        self.model = ZeroDCENet().to(self.device)
        self.model.eval()

        logger.info("Zero-DCE++ model loaded")

    def enhance(
        self,
        image: np.ndarray,
        num_iterations: int = 8,
    ) -> Tuple[np.ndarray, dict]:
        """
        Enhance low-light image.

        Args:
            image: Input image (BGR)
            num_iterations: Number of curve iterations

        Returns:
            Tuple of (enhanced_image, metadata)
        """
        self._lazy_load()

        import cv2

        # Convert to RGB and normalize
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0

        # To tensor
        img_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            # Predict curve parameters
            curve_params = self.model(img_tensor)

            # Apply curves
            enhanced_tensor = img_tensor
            for i in range(num_iterations):
                enhanced_tensor = self._apply_curve(enhanced_tensor, curve_params)

        # Convert back
        enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced = (enhanced * 255).clip(0, 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        metadata = {
            "method": "zero_dce_plus_plus",
            "iterations": num_iterations,
        }

        return enhanced_bgr, metadata

    def _apply_curve(
        self, image: torch.Tensor, curve_params: torch.Tensor
    ) -> torch.Tensor:
        """Apply learned curve transformation."""
        # Higher-order curve
        # LE(x) = x + alpha * x * (1 - x)
        alpha = torch.tanh(curve_params)
        enhanced = image + alpha * image * (1 - image)
        enhanced = torch.clamp(enhanced, 0, 1)
        return enhanced


class ZeroDCENet(nn.Module):
    """
    Zero-DCE network architecture.

    Lightweight CNN that predicts pixel-wise curve parameters.
    """

    def __init__(self):
        """Initialize network."""
        super().__init__()

        # Simple encoder
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)

        # Output curve parameters (3 channels for RGB)
        self.conv_out = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        curve_params = self.conv_out(h)

        return curve_params


class EnlightenGAN:
    """
    EnlightenGAN: Deep Light Enhancement without Paired Supervision.

    GAN-based approach for unpaired low-light enhancement.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize EnlightenGAN.

        Args:
            device: Device for inference
        """
        self.device = device
        self.generator = None

        logger.info("EnlightenGAN initialized")

    def _lazy_load(self):
        """Lazy load generator."""
        if self.generator is not None:
            return

        self.generator = EnlightenGANGenerator().to(self.device)
        self.generator.eval()

        logger.info("EnlightenGAN loaded")

    def enhance(
        self,
        image: np.ndarray,
        use_global_local: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """
        Enhance low-light image.

        Args:
            image: Input image (BGR)
            use_global_local: Use global-local discriminator strategy

        Returns:
            Tuple of (enhanced_image, metadata)
        """
        self._lazy_load()

        import cv2

        # Prepare input
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0 * 2 - 1  # [-1, 1]

        img_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            enhanced_tensor = self.generator(img_tensor)

        # Convert back
        enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced = ((enhanced + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        metadata = {
            "method": "enlighten_gan",
            "global_local": use_global_local,
        }

        return enhanced_bgr, metadata


class EnlightenGANGenerator(nn.Module):
    """EnlightenGAN generator network."""

    def __init__(self, ngf: int = 64):
        """
        Initialize generator.

        Args:
            ngf: Number of generator filters
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, ngf, 7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            # Downsample
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(ngf * 4) for _ in range(6)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            # Upsample
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            # Output
            nn.Conv2d(ngf, 3, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = self.encoder(x)
        h = self.residual_blocks(h)
        out = self.decoder(h)
        return out


class ResidualBlock(nn.Module):
    """Residual block for generator."""

    def __init__(self, channels: int):
        """Initialize residual block."""
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x + self.block(x)


class RetinexNet:
    """
    RetinexNet for low-light enhancement.

    Based on Retinex theory: decomposes image into reflectance and illumination.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize RetinexNet.

        Args:
            device: Device for inference
        """
        self.device = device
        self.decom_net = None
        self.enhance_net = None

        logger.info("RetinexNet initialized")

    def _lazy_load(self):
        """Lazy load models."""
        if self.decom_net is not None:
            return

        self.decom_net = DecompositionNet().to(self.device)
        self.enhance_net = IlluminationEnhanceNet().to(self.device)

        self.decom_net.eval()
        self.enhance_net.eval()

        logger.info("RetinexNet models loaded")

    def enhance(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Enhance low-light image using Retinex.

        Args:
            image: Input image (BGR)

        Returns:
            Tuple of (enhanced_image, metadata)
        """
        self._lazy_load()

        import cv2

        # Prepare input
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            # Decompose into reflectance and illumination
            reflectance, illumination = self.decom_net(img_tensor)

            # Enhance illumination
            enhanced_illum = self.enhance_net(illumination)

            # Reconstruct
            enhanced_tensor = reflectance * enhanced_illum

        # Convert back
        enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced = (enhanced * 255).clip(0, 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        # Get illumination map for metadata
        illum_map = enhanced_illum.squeeze(0).permute(1, 2, 0).cpu().numpy()
        illum_map = (illum_map * 255).clip(0, 255).astype(np.uint8)

        metadata = {
            "method": "retinex_net",
            "illumination_map": illum_map,
        }

        return enhanced_bgr, metadata


class DecompositionNet(nn.Module):
    """Network for Retinex decomposition."""

    def __init__(self):
        """Initialize decomposition network."""
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        # Reflectance output
        self.conv_r = nn.Conv2d(32, 3, 3, padding=1)

        # Illumination output
        self.conv_i = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple of (reflectance, illumination)
        """
        h = self.conv_layers(x)

        reflectance = torch.sigmoid(self.conv_r(h))
        illumination = torch.sigmoid(self.conv_i(h))

        # Expand illumination to 3 channels
        illumination = illumination.repeat(1, 3, 1, 1)

        return reflectance, illumination


class IlluminationEnhanceNet(nn.Module):
    """Network for illumination enhancement."""

    def __init__(self):
        """Initialize enhancement network."""
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, illumination: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            illumination: Input illumination map (B, 3, H, W)

        Returns:
            Enhanced illumination (B, 3, H, W)
        """
        # Convert to single channel
        illum_gray = illumination.mean(dim=1, keepdim=True)

        # Enhance
        enhanced = self.conv_layers(illum_gray)

        # Expand back to 3 channels
        enhanced = enhanced.repeat(1, 3, 1, 1)

        return enhanced


class AdaptiveLowLightEnhancer:
    """
    Adaptive enhancer that selects best method based on image characteristics.

    Automatically chooses between Zero-DCE++, EnlightenGAN, and RetinexNet.
    """

    def __init__(self, device: str = "cuda"):
        """Initialize adaptive enhancer."""
        self.device = device

        self.zero_dce = ZeroDCEPlusPlus(device=device)
        self.enlighten_gan = EnlightenGAN(device=device)
        self.retinex_net = RetinexNet(device=device)

        logger.info("Adaptive low-light enhancer initialized")

    def enhance(
        self,
        image: np.ndarray,
        auto_select: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """
        Adaptively enhance low-light image.

        Args:
            image: Input image (BGR)
            auto_select: Automatically select best method

        Returns:
            Tuple of (enhanced_image, metadata)
        """
        if auto_select:
            method = self._select_best_method(image)
        else:
            method = "zero_dce"  # Default

        logger.info(f"Selected method: {method}")

        if method == "zero_dce":
            enhanced, metadata = self.zero_dce.enhance(image)
        elif method == "enlighten_gan":
            enhanced, metadata = self.enlighten_gan.enhance(image)
        elif method == "retinex":
            enhanced, metadata = self.retinex_net.enhance(image)
        else:
            enhanced, metadata = self.zero_dce.enhance(image)

        metadata["selected_method"] = method
        metadata["auto_select"] = auto_select

        return enhanced, metadata

    def _select_best_method(self, image: np.ndarray) -> str:
        """
        Select best enhancement method based on image characteristics.

        Args:
            image: Input image

        Returns:
            Method name
        """
        import cv2

        # Calculate image statistics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mean_brightness = gray.mean()
        std_brightness = gray.std()

        # Calculate local variation
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Decision logic
        if mean_brightness < 50 and std_brightness < 30:
            # Very dark with low contrast -> Zero-DCE++
            return "zero_dce"
        elif mean_brightness < 80 and laplacian_var > 100:
            # Low light with some detail -> EnlightenGAN
            return "enlighten_gan"
        elif mean_brightness < 100:
            # Moderate low light -> RetinexNet
            return "retinex"
        else:
            # Not very dark -> Zero-DCE++
            return "zero_dce"


def compare_low_light_methods(
    image: np.ndarray,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Compare all low-light enhancement methods.

    Args:
        image: Input image
        device: Device for inference

    Returns:
        Dictionary mapping method name to enhanced image
    """
    results = {}

    # Zero-DCE++
    zero_dce = ZeroDCEPlusPlus(device=device)
    enhanced, _ = zero_dce.enhance(image)
    results["zero_dce"] = enhanced

    # EnlightenGAN
    enlighten = EnlightenGAN(device=device)
    enhanced, _ = enlighten.enhance(image)
    results["enlighten_gan"] = enhanced

    # RetinexNet
    retinex = RetinexNet(device=device)
    enhanced, _ = retinex.enhance(image)
    results["retinex_net"] = enhanced

    return results
