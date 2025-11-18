"""Model ensemble for improved enhancement quality."""

from typing import List, Tuple
import numpy as np
import cv2
from loguru import logger

from face_enhancement.models.face_enhancer import FaceEnhancer


class EnhancementEnsemble:
    """
    Ensemble multiple enhancement models for better results.

    Combines outputs from multiple models using various fusion strategies:
    - Weighted average
    - Quality-based selection
    - Pixel-wise maximum
    - Learned fusion (if weights provided)
    """

    def __init__(
        self,
        models: List[FaceEnhancer],
        fusion_method: str = "weighted_average",
        weights: List[float] = None,
    ):
        """
        Initialize ensemble.

        Args:
            models: List of face enhancer models
            fusion_method: Fusion method ('weighted_average', 'quality_select', 'pixel_max')
            weights: Model weights for fusion (default: equal weights)
        """
        self.models = models
        self.fusion_method = fusion_method

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

        logger.info(f"Initialized ensemble with {len(models)} models, fusion: {fusion_method}")

    def enhance(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Enhance image using ensemble.

        Args:
            image: Input image

        Returns:
            Tuple of (enhanced_image, metadata)
        """
        # Get predictions from all models
        predictions = []
        metadata = {"models": [], "fusion_method": self.fusion_method}

        for i, model in enumerate(self.models):
            logger.debug(f"Running model {i+1}/{len(self.models)}")
            enhanced, _ = model.enhance(image)
            predictions.append(enhanced)
            metadata["models"].append(model.get_model_info())

        # Fuse predictions
        if self.fusion_method == "weighted_average":
            result = self._weighted_average_fusion(predictions)
        elif self.fusion_method == "quality_select":
            result = self._quality_based_selection(predictions, image)
        elif self.fusion_method == "pixel_max":
            result = self._pixel_max_fusion(predictions)
        else:
            logger.warning(f"Unknown fusion method: {self.fusion_method}, using weighted average")
            result = self._weighted_average_fusion(predictions)

        return result, metadata

    def _weighted_average_fusion(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Weighted average fusion."""
        result = np.zeros_like(predictions[0], dtype=np.float32)

        for pred, weight in zip(predictions, self.weights):
            result += pred.astype(np.float32) * weight

        return np.clip(result, 0, 255).astype(np.uint8)

    def _quality_based_selection(
        self, predictions: List[np.ndarray], reference: np.ndarray
    ) -> np.ndarray:
        """Select best prediction based on quality metrics."""
        best_score = -1
        best_pred = predictions[0]

        for pred in predictions:
            # Calculate quality score (PSNR-like metric)
            score = self._calculate_quality_score(pred, reference)

            if score > best_score:
                best_score = score
                best_pred = pred

        logger.debug(f"Selected prediction with quality score: {best_score:.2f}")
        return best_pred

    def _pixel_max_fusion(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Take maximum value per pixel across all predictions."""
        stacked = np.stack(predictions, axis=0)
        result = np.max(stacked, axis=0)
        return result.astype(np.uint8)

    def _calculate_quality_score(
        self, enhanced: np.ndarray, reference: np.ndarray
    ) -> float:
        """
        Calculate quality score for enhanced image.

        Combines multiple metrics:
        - Sharpness (Laplacian variance)
        - Brightness consistency
        - Color naturalness
        """
        # Sharpness
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)

        # Brightness consistency
        ref_brightness = np.mean(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY))
        enh_brightness = np.mean(gray)
        brightness_score = 1.0 / (1.0 + abs(ref_brightness - enh_brightness) / 255.0)

        # Combined score
        score = sharpness * brightness_score

        return float(score)

    def get_ensemble_info(self) -> dict:
        """Get information about the ensemble."""
        return {
            "num_models": len(self.models),
            "fusion_method": self.fusion_method,
            "weights": self.weights,
            "models": [model.get_model_info() for model in self.models],
        }
