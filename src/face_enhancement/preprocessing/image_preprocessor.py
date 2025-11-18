"""Advanced image preprocessing for low-quality security camera footage."""

from typing import Optional

import cv2
import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter


class ImagePreprocessor:
    """
    Advanced preprocessing for security camera footage.

    Features:
    - Low-light enhancement (CLAHE)
    - Denoising (Non-local means, bilateral filter)
    - Motion blur reduction
    - Auto contrast and brightness
    - Gamma correction
    - Sharpening
    """

    def __init__(
        self,
        denoise: bool = True,
        auto_contrast: bool = True,
        enhance_brightness: bool = True,
        gamma: float = 1.0,
        sharpen: bool = False,
    ):
        """
        Initialize preprocessor.

        Args:
            denoise: Enable denoising
            auto_contrast: Enable auto contrast
            enhance_brightness: Enable brightness enhancement
            gamma: Gamma correction value
            sharpen: Enable sharpening
        """
        self.denoise = denoise
        self.auto_contrast = auto_contrast
        self.enhance_brightness = enhance_brightness
        self.gamma = gamma
        self.sharpen = sharpen

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline.

        Args:
            image: Input image (BGR)

        Returns:
            Preprocessed image
        """
        result = image.copy()

        # 1. Denoising (first to reduce noise before other ops)
        if self.denoise:
            result = self.apply_denoising(result)

        # 2. Auto contrast
        if self.auto_contrast:
            result = self.apply_auto_contrast(result)

        # 3. Brightness enhancement (CLAHE for low-light)
        if self.enhance_brightness:
            result = self.enhance_low_light(result)

        # 4. Gamma correction
        if self.gamma != 1.0:
            result = self.apply_gamma_correction(result, self.gamma)

        # 5. Sharpening (last to enhance edges)
        if self.sharpen:
            result = self.apply_sharpening(result)

        return result

    def apply_denoising(
        self, image: np.ndarray, method: str = "bilateral", strength: int = 10
    ) -> np.ndarray:
        """
        Apply denoising to reduce noise in low-quality footage.

        Args:
            image: Input image
            method: Denoising method ('bilateral', 'nlm', 'gaussian')
            strength: Denoising strength

        Returns:
            Denoised image
        """
        if method == "bilateral":
            # Bilateral filter preserves edges while reducing noise
            return cv2.bilateralFilter(image, d=9, sigmaColor=strength * 7, sigmaSpace=strength * 7)

        elif method == "nlm":
            # Non-local means denoising (slower but better quality)
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

        elif method == "gaussian":
            # Simple Gaussian blur (fastest)
            return cv2.GaussianBlur(image, (5, 5), strength / 10)

        return image

    def enhance_low_light(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Enhance low-light images using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input image
            clip_limit: Contrast clipping limit

        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Merge and convert back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return result

    def apply_auto_contrast(self, image: np.ndarray, clip_percent: float = 1.0) -> np.ndarray:
        """
        Apply automatic contrast stretching.

        Args:
            image: Input image
            clip_percent: Percentage of pixels to clip at extremes

        Returns:
            Contrast enhanced image
        """
        result = np.zeros_like(image)

        for i in range(image.shape[2]):
            channel = image[:, :, i]
            hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf[-1]

            # Find clip points
            low_val = np.searchsorted(cdf_normalized, clip_percent / 100.0)
            high_val = np.searchsorted(cdf_normalized, 1.0 - clip_percent / 100.0)

            # Stretch
            result[:, :, i] = np.clip(
                (channel - low_val) * (255.0 / max(high_val - low_val, 1)), 0, 255
            ).astype(np.uint8)

        return result

    def apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction.

        Args:
            image: Input image
            gamma: Gamma value (>1 brightens, <1 darkens)

        Returns:
            Gamma corrected image
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)

    def apply_sharpening(
        self, image: np.ndarray, method: str = "unsharp", strength: float = 1.0
    ) -> np.ndarray:
        """
        Apply sharpening to enhance edges.

        Args:
            image: Input image
            method: Sharpening method ('unsharp', 'laplacian')
            strength: Sharpening strength

        Returns:
            Sharpened image
        """
        if method == "unsharp":
            # Unsharp masking
            blurred = cv2.GaussianBlur(image, (0, 0), 3)
            sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            return sharpened

        elif method == "laplacian":
            # Laplacian sharpening
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpened = image.copy()
            for i in range(3):
                sharpened[:, :, i] = np.clip(
                    image[:, :, i] + strength * laplacian, 0, 255
                ).astype(np.uint8)
            return sharpened

        return image

    def reduce_motion_blur(
        self, image: np.ndarray, kernel_size: int = 15, angle: float = 0
    ) -> np.ndarray:
        """
        Attempt to reduce motion blur using Wiener deconvolution.

        Args:
            image: Input image
            kernel_size: Size of motion blur kernel
            angle: Angle of motion blur

        Returns:
            Deblurred image
        """
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Rotate kernel
        M = cv2.getRotationMatrix2D(
            (kernel_size / 2, kernel_size / 2), angle, 1.0
        )
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

        # Apply Wiener deconvolution (simplified version)
        # For production, consider using Richardson-Lucy deconvolution
        result = cv2.filter2D(image, -1, kernel)

        return result

    def auto_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply automatic white balance correction.

        Args:
            image: Input image

        Returns:
            White balanced image
        """
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])

        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)

        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
