"""Progressive multi-stage face enhancement for extreme quality."""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from loguru import logger

from face_enhancement.models.face_enhancer import FaceEnhancer
from face_enhancement.preprocessing.image_preprocessor import ImagePreprocessor


class ProgressiveEnhancer:
    """
    Multi-stage progressive enhancement pipeline.

    Applies enhancement in multiple stages with increasing quality:
    Stage 1: Basic preprocessing and denoising
    Stage 2: Initial enhancement at lower resolution
    Stage 3: Fine-grained enhancement at target resolution
    Stage 4: Detail refinement and sharpening

    This approach allows for better convergence and higher quality results,
    especially for severely degraded images.
    """

    def __init__(
        self,
        base_model: str = "gfpgan",
        stages: int = 3,
        device: str = "cuda",
    ):
        """
        Initialize progressive enhancer.

        Args:
            base_model: Base enhancement model
            stages: Number of progressive stages (2-4 recommended)
            device: Processing device
        """
        self.base_model = base_model
        self.stages = stages
        self.device = device

        # Initialize components
        self.preprocessor = ImagePreprocessor(
            denoise=True,
            auto_contrast=True,
            enhance_brightness=True,
        )

        # Create enhancers for different stages
        self.enhancers = self._create_stage_enhancers()

        logger.info(f"Initialized progressive enhancer with {stages} stages")

    def _create_stage_enhancers(self) -> List[FaceEnhancer]:
        """Create enhancers for each stage."""
        enhancers = []

        for i in range(self.stages):
            # Progressive upscaling: 1x -> 2x -> 4x
            upscale = min(2 ** (i + 1), 4) if self.stages > 1 else 2

            enhancer = FaceEnhancer(
                model=self.base_model,
                upscale=upscale,
                device=self.device,
                fidelity_weight=0.3 + (i * 0.2),  # Increase fidelity progressively
            )
            enhancers.append(enhancer)

        return enhancers

    def enhance(
        self,
        image: np.ndarray,
        intermediate_outputs: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply progressive enhancement.

        Args:
            image: Input image
            intermediate_outputs: Return all intermediate stage outputs

        Returns:
            Tuple of (final_enhanced, metadata)
        """
        logger.info("Starting progressive enhancement")

        metadata = {
            "stages": self.stages,
            "stage_outputs": [] if intermediate_outputs else None,
            "original_shape": image.shape,
        }

        # Stage 0: Preprocessing
        logger.debug("Stage 0: Preprocessing")
        current = self.preprocessor.preprocess(image)

        if intermediate_outputs:
            metadata["stage_outputs"].append(("preprocessing", current.copy()))

        # Progressive enhancement stages
        for i, enhancer in enumerate(self.enhancers):
            logger.debug(f"Stage {i+1}: Enhancement (upscale={enhancer.upscale})")

            # Enhance
            current, _ = enhancer.enhance(current)

            if intermediate_outputs:
                metadata["stage_outputs"].append((f"stage_{i+1}", current.copy()))

            # Apply intermediate refinements
            if i < len(self.enhancers) - 1:
                current = self._apply_intermediate_refinement(current, stage=i)

        # Final refinement
        logger.debug("Final stage: Refinement")
        final = self._apply_final_refinement(current)

        if intermediate_outputs:
            metadata["stage_outputs"].append(("final_refinement", final.copy()))

        metadata["final_shape"] = final.shape

        logger.info("Progressive enhancement complete")

        return final, metadata

    def _apply_intermediate_refinement(
        self, image: np.ndarray, stage: int
    ) -> np.ndarray:
        """
        Apply refinement between stages.

        Args:
            image: Image to refine
            stage: Current stage number

        Returns:
            Refined image
        """
        # Light denoising to remove artifacts
        refined = cv2.bilateralFilter(image, d=5, sigmaColor=10, sigmaSpace=10)

        # Edge-preserving smoothing
        refined = cv2.edgePreservingFilter(refined, flags=cv2.RECURS_FILTER, sigma_s=20, sigma_r=0.1)

        return refined

    def _apply_final_refinement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply final refinement to enhanced image.

        Args:
            image: Enhanced image

        Returns:
            Refined image
        """
        # Convert to LAB for perceptual refinement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Gentle sharpening on luminance
        blurred = cv2.GaussianBlur(l, (0, 0), 1.0)
        l_sharp = cv2.addWeighted(l, 1.5, blurred, -0.5, 0)
        l_sharp = np.clip(l_sharp, 0, 255).astype(np.uint8)

        # Reduce color noise
        a_smooth = cv2.bilateralFilter(a, d=5, sigmaColor=10, sigmaSpace=10)
        b_smooth = cv2.bilateralFilter(b, d=5, sigmaColor=10, sigmaSpace=10)

        # Reconstruct
        lab_refined = cv2.merge([l_sharp, a_smooth, b_smooth])
        refined = cv2.cvtColor(lab_refined, cv2.COLOR_LAB2BGR)

        return refined

    def enhance_with_guidance(
        self,
        image: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Enhance with optional reference image for guidance.

        Args:
            image: Input image to enhance
            reference: Optional reference image for style/quality guidance

        Returns:
            Tuple of (enhanced, metadata)
        """
        if reference is None:
            return self.enhance(image)

        logger.info("Enhancing with reference guidance")

        # Extract style features from reference
        ref_stats = self._extract_style_stats(reference)

        # Enhance normally
        enhanced, metadata = self.enhance(image)

        # Apply guided refinement
        enhanced = self._apply_guided_refinement(enhanced, ref_stats)

        metadata["guided"] = True

        return enhanced, metadata

    def _extract_style_stats(self, reference: np.ndarray) -> dict:
        """Extract style statistics from reference image."""
        lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        return {
            "l_mean": np.mean(l),
            "l_std": np.std(l),
            "a_mean": np.mean(a),
            "a_std": np.std(a),
            "b_mean": np.mean(b),
            "b_std": np.std(b),
        }

    def _apply_guided_refinement(
        self, image: np.ndarray, ref_stats: dict
    ) -> np.ndarray:
        """Apply style transfer based on reference statistics."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Match color statistics to reference
        l_matched = self._match_stats(l, ref_stats["l_mean"], ref_stats["l_std"])
        a_matched = self._match_stats(a, ref_stats["a_mean"], ref_stats["a_std"])
        b_matched = self._match_stats(b, ref_stats["b_mean"], ref_stats["b_std"])

        lab_matched = cv2.merge([l_matched, a_matched, b_matched])
        result = cv2.cvtColor(lab_matched, cv2.COLOR_LAB2BGR)

        return result

    def _match_stats(
        self, channel: np.ndarray, target_mean: float, target_std: float
    ) -> np.ndarray:
        """Match channel statistics to target."""
        current_mean = np.mean(channel)
        current_std = np.std(channel)

        # Normalize and rescale
        normalized = (channel - current_mean) / (current_std + 1e-6)
        matched = normalized * target_std + target_mean

        return np.clip(matched, 0, 255).astype(np.uint8)

    def get_stage_info(self) -> dict:
        """Get information about enhancement stages."""
        return {
            "num_stages": self.stages,
            "base_model": self.base_model,
            "stage_upscales": [e.upscale for e in self.enhancers],
            "stage_fidelity": [e.fidelity_weight for e in self.enhancers],
        }
