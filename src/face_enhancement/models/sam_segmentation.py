"""Segment Anything Model (SAM) integration for precise face segmentation.

Uses Meta's SAM for:
- Precise face boundary detection
- Multi-region face segmentation
- Automatic mask generation
- Interactive segmentation refinement
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
import cv2
import torch
from loguru import logger


class SAMFaceSegmenter:
    """
    Segment Anything Model for face segmentation.

    Provides pixel-perfect face masks for:
    - Face region isolation
    - Hair/background separation
    - Facial component segmentation
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize SAM segmenter.

        Args:
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: Path to SAM checkpoint
            device: Device for inference
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.sam = None
        self.predictor = None
        self.mask_generator = None

        logger.info(f"SAM Face Segmenter initialized ({model_type})")

    def _lazy_load(self):
        """Lazy load SAM model."""
        if self.sam is not None:
            return

        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

            # Load SAM model
            if self.checkpoint_path:
                self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            else:
                # Try to load from default location
                logger.warning("No checkpoint path provided. Using placeholder.")
                self.sam = self._create_placeholder_sam()

            self.sam = self.sam.to(self.device)

            # Create predictor and mask generator
            self.predictor = SamPredictor(self.sam) if hasattr(self.sam, 'image_encoder') else None
            self.mask_generator = SamAutomaticMaskGenerator(self.sam) if hasattr(self.sam, 'image_encoder') else None

            logger.info("SAM model loaded successfully")

        except ImportError:
            logger.warning("segment-anything not installed. Using fallback segmentation.")
            self.sam = self._create_placeholder_sam()

    def _create_placeholder_sam(self):
        """Create placeholder SAM for when library is not available."""
        class PlaceholderSAM:
            def __init__(self):
                self.image_encoder = None

        return PlaceholderSAM()

    def segment_face(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        return_multiple_masks: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Segment face from image.

        Args:
            image: Input image (BGR)
            face_bbox: Optional face bounding box (x1, y1, x2, y2)
            return_multiple_masks: Return multiple candidate masks

        Returns:
            Tuple of (mask, metadata)
        """
        self._lazy_load()

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if face_bbox is not None:
            # Use bbox to guide segmentation
            mask, metadata = self._segment_with_box(image_rgb, face_bbox)
        else:
            # Automatic face detection and segmentation
            mask, metadata = self._auto_segment_face(image_rgb)

        return mask, metadata

    def _segment_with_box(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, Dict]:
        """Segment face using bounding box prompt."""
        if self.predictor is None:
            return self._fallback_segment_with_box(image, bbox)

        # Set image
        self.predictor.set_image(image)

        # Convert bbox to SAM format
        input_box = np.array(bbox)

        # Predict masks
        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=True,
        )

        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        metadata = {
            "method": "sam_box",
            "confidence": float(scores[best_idx]),
            "num_candidates": len(masks),
        }

        return mask.astype(np.uint8) * 255, metadata

    def _auto_segment_face(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Automatically segment face."""
        if self.mask_generator is None:
            return self._fallback_auto_segment(image)

        # Generate masks
        masks = self.mask_generator.generate(image)

        # Find face mask (largest mask in center)
        h, w = image.shape[:2]
        center_y, center_x = h // 2, w // 2

        best_mask = None
        best_score = 0

        for mask_data in masks:
            mask = mask_data['segmentation']
            area = mask_data['area']

            # Check if center is in mask
            if mask[center_y, center_x]:
                # Score based on area and predicted IoU
                score = area * mask_data.get('predicted_iou', 0.5)

                if score > best_score:
                    best_score = score
                    best_mask = mask

        if best_mask is None:
            # Fallback: use largest mask
            best_mask = max(masks, key=lambda m: m['area'])['segmentation']

        metadata = {
            "method": "sam_auto",
            "num_masks_found": len(masks),
            "score": float(best_score),
        }

        return best_mask.astype(np.uint8) * 255, metadata

    def _fallback_segment_with_box(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, Dict]:
        """Fallback segmentation using GrabCut."""
        x1, y1, x2, y2 = bbox

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        rect = (x1, y1, x2 - x1, y2 - y1)

        # Run GrabCut
        cv2.grabCut(
            image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT
        )

        # Get foreground mask
        mask_fg = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        mask_fg = mask_fg * 255

        metadata = {
            "method": "grabcut_fallback",
            "confidence": 0.7,
        }

        return mask_fg, metadata

    def _fallback_auto_segment(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Fallback auto segmentation using simple methods."""
        # Use face detection + GrabCut
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect face
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            bbox = (x, y, x + w, y + h)
            return self._fallback_segment_with_box(image, bbox)
        else:
            # Return full image mask
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            return mask, {"method": "full_image_fallback"}

    def segment_face_components(
        self, image: np.ndarray, face_landmarks: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Segment individual face components.

        Args:
            image: Face image
            face_landmarks: Facial landmarks (68 points)

        Returns:
            Dictionary of component masks
        """
        components = {}

        # Define component point ranges (68-landmark model)
        component_ranges = {
            "left_eye": (36, 42),
            "right_eye": (42, 48),
            "nose": (27, 36),
            "mouth": (48, 68),
            "left_eyebrow": (17, 22),
            "right_eyebrow": (22, 27),
            "jaw": (0, 17),
        }

        h, w = image.shape[:2]

        for component_name, (start_idx, end_idx) in component_ranges.items():
            # Get component landmarks
            points = face_landmarks[start_idx:end_idx]

            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 255)

            # Optionally refine with SAM
            if self.predictor is not None:
                # Use points as prompt
                # (Simplified - real implementation would use SAM point prompts)
                pass

            components[component_name] = mask

        return components

    def refine_mask_interactive(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray,
        positive_points: Optional[List[Tuple[int, int]]] = None,
        negative_points: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Refine mask interactively with point prompts.

        Args:
            image: Input image
            initial_mask: Initial mask
            positive_points: Points that should be included
            negative_points: Points that should be excluded

        Returns:
            Refined mask
        """
        if self.predictor is None:
            return initial_mask

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        # Prepare point prompts
        points = []
        labels = []

        if positive_points:
            for pt in positive_points:
                points.append(pt)
                labels.append(1)

        if negative_points:
            for pt in negative_points:
                points.append(pt)
                labels.append(0)

        if not points:
            return initial_mask

        point_coords = np.array(points)
        point_labels = np.array(labels)

        # Predict with points
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=None,
            multimask_output=True,
        )

        # Select best mask
        best_idx = np.argmax(scores)
        refined_mask = masks[best_idx].astype(np.uint8) * 255

        return refined_mask

    def apply_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        background: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply mask to image.

        Args:
            image: Input image
            mask: Binary mask
            background: Optional background image

        Returns:
            Masked image
        """
        # Normalize mask
        mask_norm = mask.astype(np.float32) / 255.0

        if len(mask_norm.shape) == 2:
            mask_norm = mask_norm[:, :, None]

        if background is not None:
            # Composite with background
            result = image * mask_norm + background * (1 - mask_norm)
        else:
            # Use black background
            result = image * mask_norm

        return result.astype(np.uint8)


class SAMEnhancementIntegration:
    """
    Integration of SAM with face enhancement.

    Uses precise SAM masks to:
    - Isolate face for enhancement
    - Preserve background
    - Blend enhanced face smoothly
    """

    def __init__(self, sam_segmenter: SAMFaceSegmenter):
        """
        Initialize integration.

        Args:
            sam_segmenter: SAM segmenter instance
        """
        self.sam_segmenter = sam_segmenter
        logger.info("SAM Enhancement Integration initialized")

    def enhance_with_sam_mask(
        self,
        image: np.ndarray,
        enhancer_fn,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        blend_mode: str = "poisson",
    ) -> Tuple[np.ndarray, Dict]:
        """
        Enhance face using SAM for precise masking.

        Args:
            image: Input image
            enhancer_fn: Enhancement function
            face_bbox: Optional face bounding box
            blend_mode: Blending mode ('alpha', 'poisson', 'pyramid')

        Returns:
            Tuple of (enhanced_image, metadata)
        """
        # Get precise face mask
        mask, mask_metadata = self.sam_segmenter.segment_face(image, face_bbox)

        # Extract face region
        face_region = cv2.bitwise_and(image, image, mask=mask)

        # Enhance face
        enhanced_face = enhancer_fn(face_region)

        # Blend back
        if blend_mode == "poisson":
            result = self._poisson_blend(image, enhanced_face, mask)
        elif blend_mode == "pyramid":
            result = self._pyramid_blend(image, enhanced_face, mask)
        else:
            result = self._alpha_blend(image, enhanced_face, mask)

        metadata = {
            "segmentation": mask_metadata,
            "blend_mode": blend_mode,
        }

        return result, metadata

    def _alpha_blend(
        self, background: np.ndarray, foreground: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Alpha blending."""
        # Feather mask edges
        mask_blur = cv2.GaussianBlur(mask, (21, 21), 11)
        mask_norm = mask_blur.astype(np.float32) / 255.0

        if len(mask_norm.shape) == 2:
            mask_norm = mask_norm[:, :, None]

        result = foreground * mask_norm + background * (1 - mask_norm)
        return result.astype(np.uint8)

    def _poisson_blend(
        self, background: np.ndarray, foreground: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Poisson blending for seamless compositing."""
        # Find center of mask
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            center = (cx, cy)
        else:
            h, w = mask.shape
            center = (w // 2, h // 2)

        # Poisson blending
        try:
            result = cv2.seamlessClone(
                foreground, background, mask, center, cv2.NORMAL_CLONE
            )
        except:
            # Fallback to alpha blend
            result = self._alpha_blend(background, foreground, mask)

        return result

    def _pyramid_blend(
        self, background: np.ndarray, foreground: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Pyramid blending for multi-scale compositing."""
        # Simplified pyramid blending
        num_levels = 5

        # Build Gaussian pyramids
        bg_pyramid = [background]
        fg_pyramid = [foreground]
        mask_pyramid = [mask.astype(np.float32) / 255.0]

        for i in range(num_levels - 1):
            bg_pyramid.append(cv2.pyrDown(bg_pyramid[-1]))
            fg_pyramid.append(cv2.pyrDown(fg_pyramid[-1]))
            mask_pyramid.append(cv2.pyrDown(mask_pyramid[-1]))

        # Build Laplacian pyramids
        bg_laplacian = []
        fg_laplacian = []

        for i in range(num_levels - 1):
            size = (bg_pyramid[i].shape[1], bg_pyramid[i].shape[0])
            expanded = cv2.pyrUp(bg_pyramid[i + 1], dstsize=size)
            bg_laplacian.append(bg_pyramid[i] - expanded)

            expanded = cv2.pyrUp(fg_pyramid[i + 1], dstsize=size)
            fg_laplacian.append(fg_pyramid[i] - expanded)

        bg_laplacian.append(bg_pyramid[-1])
        fg_laplacian.append(fg_pyramid[-1])

        # Blend in Laplacian pyramid
        blended_pyramid = []
        for i in range(num_levels):
            mask_layer = mask_pyramid[i]
            if len(mask_layer.shape) == 2:
                mask_layer = mask_layer[:, :, None]

            blended = fg_laplacian[i] * mask_layer + bg_laplacian[i] * (1 - mask_layer)
            blended_pyramid.append(blended)

        # Reconstruct
        result = blended_pyramid[-1]
        for i in range(num_levels - 2, -1, -1):
            size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
            result = cv2.pyrUp(result, dstsize=size)
            result = result + blended_pyramid[i]

        return np.clip(result, 0, 255).astype(np.uint8)
