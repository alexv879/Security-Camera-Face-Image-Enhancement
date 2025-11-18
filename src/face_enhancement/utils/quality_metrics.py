"""Image quality assessment metrics."""

import cv2
import numpy as np
from typing import Dict, Tuple
from scipy import signal
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class QualityAssessment:
    """
    Comprehensive image quality assessment.

    Metrics:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
    - Sharpness
    - Brightness
    - Contrast
    - Colorfulness
    """

    @staticmethod
    def calculate_psnr(reference: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Calculate PSNR between reference and enhanced images.

        Args:
            reference: Reference image
            enhanced: Enhanced image

        Returns:
            PSNR value in dB
        """
        return float(psnr(reference, enhanced))

    @staticmethod
    def calculate_ssim(reference: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Calculate SSIM between reference and enhanced images.

        Args:
            reference: Reference image
            enhanced: Enhanced image

        Returns:
            SSIM value (0-1)
        """
        # Convert to grayscale if needed
        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference
            enh_gray = enhanced

        return float(ssim(ref_gray, enh_gray))

    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.

        Args:
            image: Input image

        Returns:
            Sharpness score (higher is sharper)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(laplacian))

    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """
        Calculate average brightness.

        Args:
            image: Input image

        Returns:
            Average brightness (0-255)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        return float(np.mean(gray))

    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """
        Calculate RMS contrast.

        Args:
            image: Input image

        Returns:
            RMS contrast value
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        return float(np.std(gray))

    @staticmethod
    def calculate_colorfulness(image: np.ndarray) -> float:
        """
        Calculate colorfulness metric.

        Args:
            image: Input image (BGR)

        Returns:
            Colorfulness score
        """
        if len(image.shape) != 3:
            return 0.0

        # Split into channels
        b, g, r = cv2.split(image)

        # Calculate RG and YB
        rg = np.absolute(r.astype(np.float32) - g.astype(np.float32))
        yb = np.absolute(0.5 * (r.astype(np.float32) + g.astype(np.float32)) - b.astype(np.float32))

        # Calculate std and mean
        rg_std = np.std(rg)
        rg_mean = np.mean(rg)
        yb_std = np.std(yb)
        yb_mean = np.mean(yb)

        # Calculate colorfulness
        std_root = np.sqrt(rg_std ** 2 + yb_std ** 2)
        mean_root = np.sqrt(rg_mean ** 2 + yb_mean ** 2)

        return float(std_root + 0.3 * mean_root)

    @staticmethod
    def calculate_niqe(image: np.ndarray) -> float:
        """
        Calculate NIQE (Natural Image Quality Evaluator) score.

        Note: This is a simplified version. For full NIQE, use specialized libraries.

        Args:
            image: Input image

        Returns:
            NIQE score (lower is better)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate local statistics
        mu = cv2.GaussianBlur(gray.astype(np.float32), (7, 7), 7/6)
        sigma = cv2.GaussianBlur((gray.astype(np.float32) - mu) ** 2, (7, 7), 7/6)
        sigma = np.sqrt(sigma)

        # Simplified NIQE score
        score = float(np.mean(sigma))

        return score

    @staticmethod
    def calculate_entropy(image: np.ndarray) -> float:
        """
        Calculate image entropy (information content).

        Args:
            image: Input image

        Returns:
            Entropy value
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate histogram
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))

        # Normalize
        hist = hist / hist.sum()

        # Calculate entropy
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))

        return float(entropy)

    @staticmethod
    def calculate_all_metrics(
        image: np.ndarray,
        reference: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate all quality metrics.

        Args:
            image: Enhanced image
            reference: Optional reference image for PSNR/SSIM

        Returns:
            Dict of metric name to value
        """
        metrics = {
            "sharpness": QualityAssessment.calculate_sharpness(image),
            "brightness": QualityAssessment.calculate_brightness(image),
            "contrast": QualityAssessment.calculate_contrast(image),
            "colorfulness": QualityAssessment.calculate_colorfulness(image),
            "entropy": QualityAssessment.calculate_entropy(image),
            "niqe": QualityAssessment.calculate_niqe(image),
        }

        if reference is not None:
            metrics["psnr"] = QualityAssessment.calculate_psnr(reference, image)
            metrics["ssim"] = QualityAssessment.calculate_ssim(reference, image)

        return metrics

    @staticmethod
    def compare_images(
        original: np.ndarray,
        enhanced: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare original and enhanced images.

        Args:
            original: Original image
            enhanced: Enhanced image

        Returns:
            Dict with metrics for both images and improvements
        """
        original_metrics = QualityAssessment.calculate_all_metrics(original)
        enhanced_metrics = QualityAssessment.calculate_all_metrics(enhanced, reference=original)

        improvements = {}
        for key in original_metrics.keys():
            if key in enhanced_metrics:
                improvements[key] = enhanced_metrics[key] - original_metrics[key]

        return {
            "original": original_metrics,
            "enhanced": enhanced_metrics,
            "improvement": improvements,
        }
