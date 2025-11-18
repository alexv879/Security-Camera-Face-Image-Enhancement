"""Image processing utilities."""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load image from file.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (BGR format)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return img


def save_image(image: np.ndarray, output_path: Union[str, Path], quality: int = 95) -> None:
    """
    Save image to file.

    Args:
        image: Image array (BGR format)
        output_path: Output file path
        quality: JPEG quality (1-100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in [".jpg", ".jpeg"]:
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(output_path), image)


def resize_image(
    image: np.ndarray, target_size: Optional[Tuple[int, int]] = None, scale: Optional[float] = None
) -> np.ndarray:
    """
    Resize image to target size or scale.

    Args:
        image: Input image
        target_size: Target (width, height), or None
        scale: Scale factor, or None

    Returns:
        Resized image
    """
    if target_size is not None:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    elif scale is not None:
        h, w = image.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Args:
        image: Input image (uint8)

    Returns:
        Normalized image (float32)
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [0, 1] to [0, 255].

    Args:
        image: Normalized image (float32)

    Returns:
        Image as uint8
    """
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def create_comparison_image(
    original: np.ndarray, enhanced: np.ndarray, padding: int = 10
) -> np.ndarray:
    """
    Create side-by-side comparison of original and enhanced images.

    Args:
        original: Original image
        enhanced: Enhanced image
        padding: Padding between images

    Returns:
        Comparison image
    """
    # Ensure same size
    h = max(original.shape[0], enhanced.shape[0])
    w = max(original.shape[1], enhanced.shape[1])

    original_resized = cv2.resize(original, (w, h))
    enhanced_resized = cv2.resize(enhanced, (w, h))

    # Create padding
    pad = np.ones((h, padding, 3), dtype=np.uint8) * 255

    # Concatenate horizontally
    comparison = np.hstack([original_resized, pad, enhanced_resized])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 0, 255), 2)
    cv2.putText(comparison, "Enhanced", (w + padding + 10, 30), font, 1, (0, 255, 0), 2)

    return comparison


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to image.

    Args:
        image: Input image
        gamma: Gamma value (>1 brightens, <1 darkens)

    Returns:
        Gamma corrected image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def auto_contrast(image: np.ndarray, clip_percent: float = 1.0) -> np.ndarray:
    """
    Apply automatic contrast enhancement.

    Args:
        image: Input image
        clip_percent: Percentage of pixels to clip

    Returns:
        Contrast enhanced image
    """
    result = np.zeros_like(image)
    for i in range(image.shape[2]):
        hist, bins = np.histogram(image[:, :, i].flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]

        # Find clip points
        low_idx = np.searchsorted(cdf_normalized, clip_percent / 100.0)
        high_idx = np.searchsorted(cdf_normalized, 1.0 - clip_percent / 100.0)

        # Apply stretch
        result[:, :, i] = np.clip(
            (image[:, :, i] - low_idx) * (255.0 / (high_idx - low_idx)), 0, 255
        ).astype(np.uint8)

    return result


def draw_face_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw face detection box on image.

    Args:
        image: Input image
        box: Face bounding box (x1, y1, x2, y2)
        confidence: Detection confidence
        color: Box color (BGR)
        thickness: Box line thickness

    Returns:
        Image with drawn box
    """
    result = image.copy()
    x1, y1, x2, y2 = [int(v) for v in box]

    # Draw box
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

    # Draw confidence
    label = f"{confidence:.2f}"
    cv2.putText(result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return result
