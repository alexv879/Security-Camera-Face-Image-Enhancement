"""Perceptual loss functions for quality assessment."""

import numpy as np
import cv2
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss:
    """
    Perceptual loss calculator using VGG features.

    Measures perceptual similarity between images using deep features
    from pretrained networks, which correlates better with human perception
    than pixel-wise metrics like MSE.
    """

    def __init__(self, device: str = "cuda", use_vgg: bool = True):
        """
        Initialize perceptual loss.

        Args:
            device: Device for computation
            use_vgg: Use VGG features (vs simple LPIPS)
        """
        self.device = device
        self.use_vgg = use_vgg

        if use_vgg:
            try:
                from torchvision import models
                # Load pretrained VGG16
                vgg = models.vgg16(pretrained=True).features
                self.feature_extractor = vgg.to(device).eval()

                # Layers for feature extraction
                self.feature_layers = [3, 8, 15, 22]  # conv1_2, conv2_2, conv3_3, conv4_3

                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

            except ImportError:
                use_vgg = False
                logger.warning("torchvision not available, using simple perceptual loss")

    def calculate(
        self,
        reference: np.ndarray,
        enhanced: np.ndarray,
        normalize: bool = True,
    ) -> float:
        """
        Calculate perceptual loss.

        Args:
            reference: Reference image
            enhanced: Enhanced image
            normalize: Normalize images to [0, 1]

        Returns:
            Perceptual loss value
        """
        if not self.use_vgg:
            return self._simple_perceptual_loss(reference, enhanced)

        # Convert to tensor
        ref_tensor = self._to_tensor(reference, normalize)
        enh_tensor = self._to_tensor(enhanced, normalize)

        with torch.no_grad():
            # Extract features
            ref_features = self._extract_features(ref_tensor)
            enh_features = self._extract_features(enh_tensor)

            # Calculate loss
            loss = 0.0
            for ref_feat, enh_feat in zip(ref_features, enh_features):
                loss += F.mse_loss(ref_feat, enh_feat).item()

        return loss

    def _to_tensor(self, image: np.ndarray, normalize: bool = True) -> torch.Tensor:
        """Convert numpy image to tensor."""
        # BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

        if normalize:
            tensor = tensor / 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor.to(self.device)

    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract features from multiple layers."""
        features = []
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)

        return features

    def _simple_perceptual_loss(
        self, reference: np.ndarray, enhanced: np.ndarray
    ) -> float:
        """Simple perceptual loss without neural network."""
        # Convert to LAB
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        enh_lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)

        # Calculate per-channel loss
        loss = 0.0
        weights = [1.0, 0.5, 0.5]  # Weight L channel more

        for i, w in enumerate(weights):
            diff = (ref_lab[:, :, i].astype(float) - enh_lab[:, :, i].astype(float))
            loss += w * np.mean(diff ** 2)

        return loss


class FaceFidelityLoss:
    """
    Face-specific fidelity loss.

    Focuses on preserving facial identity while enhancing quality.
    """

    def __init__(self, device: str = "cuda"):
        """Initialize face fidelity loss."""
        self.device = device

    def calculate(
        self,
        reference_face: np.ndarray,
        enhanced_face: np.ndarray,
    ) -> dict:
        """
        Calculate face fidelity metrics.

        Args:
            reference_face: Reference face crop
            enhanced_face: Enhanced face crop

        Returns:
            Dictionary of fidelity metrics
        """
        metrics = {}

        # Structural similarity in face regions
        metrics["face_ssim"] = self._calculate_face_ssim(reference_face, enhanced_face)

        # Color fidelity (skin tone preservation)
        metrics["skin_tone_error"] = self._calculate_skin_tone_error(
            reference_face, enhanced_face
        )

        # Detail preservation
        metrics["detail_score"] = self._calculate_detail_preservation(
            reference_face, enhanced_face
        )

        # Overall fidelity score
        metrics["fidelity_score"] = (
            metrics["face_ssim"] * 0.5 +
            (1.0 - metrics["skin_tone_error"]) * 0.3 +
            metrics["detail_score"] * 0.2
        )

        return metrics

    def _calculate_face_ssim(
        self, reference: np.ndarray, enhanced: np.ndarray
    ) -> float:
        """Calculate SSIM for face region."""
        from skimage.metrics import structural_similarity as ssim

        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # Resize if needed
        if ref_gray.shape != enh_gray.shape:
            enh_gray = cv2.resize(enh_gray, (ref_gray.shape[1], ref_gray.shape[0]))

        return ssim(ref_gray, enh_gray)

    def _calculate_skin_tone_error(
        self, reference: np.ndarray, enhanced: np.ndarray
    ) -> float:
        """Calculate skin tone preservation error."""
        # Convert to YCrCb (better for skin detection)
        ref_ycrcb = cv2.cvtColor(reference, cv2.COLOR_BGR2YCrCb)
        enh_ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)

        # Simple skin detection (Cr and Cb values)
        ref_mean_cr = np.mean(ref_ycrcb[:, :, 1])
        ref_mean_cb = np.mean(ref_ycrcb[:, :, 2])

        enh_mean_cr = np.mean(enh_ycrcb[:, :, 1])
        enh_mean_cb = np.mean(enh_ycrcb[:, :, 2])

        # Calculate error
        error = (
            abs(ref_mean_cr - enh_mean_cr) / 255.0 +
            abs(ref_mean_cb - enh_mean_cb) / 255.0
        ) / 2.0

        return min(error, 1.0)

    def _calculate_detail_preservation(
        self, reference: np.ndarray, enhanced: np.ndarray
    ) -> float:
        """Calculate detail preservation score."""
        # High frequency content (edges)
        ref_edges = cv2.Canny(
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), 50, 150
        )
        enh_edges = cv2.Canny(
            cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), 50, 150
        )

        # Resize if needed
        if ref_edges.shape != enh_edges.shape:
            enh_edges = cv2.resize(enh_edges, (ref_edges.shape[1], ref_edges.shape[0]))

        # Calculate overlap
        intersection = np.logical_and(ref_edges > 0, enh_edges > 0).sum()
        union = np.logical_or(ref_edges > 0, enh_edges > 0).sum()

        if union == 0:
            return 0.0

        return intersection / union
