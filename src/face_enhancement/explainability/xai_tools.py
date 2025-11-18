"""Explainable AI (XAI) tools for face enhancement.

Provides interpretability and explainability:
- Grad-CAM for attention visualization
- LIME for local explanations
- SHAP for feature importance
- Activation maximization
- Attribution maps
"""

from typing import Tuple, Optional, Dict, List
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Visualizes which regions of the image the model focuses on.
    """

    def __init__(self, model: nn.Module, target_layer: str):
        """
        Initialize Grad-CAM.

        Args:
            model: PyTorch model
            target_layer: Name of target layer for visualization
        """
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

        logger.info(f"Grad-CAM initialized for layer: {target_layer}")

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break

    def generate_cam(
        self,
        image: np.ndarray,
        target_output: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Generate CAM for image.

        Args:
            image: Input image (BGR)
            target_output: Target output for gradient (if None, uses mean)

        Returns:
            CAM heatmap (0-255)
        """
        # Prepare input
        img_tensor = self._prepare_image(image)

        # Forward pass
        self.model.zero_grad()
        output = self.model(img_tensor)

        # Backward pass
        if target_output is None:
            target_output = output.mean()

        target_output.backward()

        # Calculate CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize to input size
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Convert to heatmap
        cam_uint8 = (cam_resized * 255).astype(np.uint8)

        return cam_uint8

    def visualize_cam(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay CAM on image.

        Args:
            image: Original image
            cam: CAM heatmap
            alpha: Transparency

        Returns:
            Visualization image
        """
        # Apply colormap
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

        return overlay

    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image for model."""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

        if next(self.model.parameters()).is_cuda:
            img_tensor = img_tensor.cuda()

        return img_tensor


class LIME:
    """
    Local Interpretable Model-agnostic Explanations.

    Explains individual predictions by approximating locally with
    interpretable models.
    """

    def __init__(self):
        """Initialize LIME."""
        logger.info("LIME initialized")

    def explain_enhancement(
        self,
        image: np.ndarray,
        enhancement_fn,
        num_samples: int = 1000,
        num_features: int = 10,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Explain enhancement decision using LIME.

        Args:
            image: Input image
            enhancement_fn: Enhancement function
            num_samples: Number of perturbation samples
            num_features: Number of features to show

        Returns:
            Tuple of (explanation_image, metadata)
        """
        try:
            from lime import lime_image
            from skimage.segmentation import mark_boundaries

            # Create LIME explainer
            explainer = lime_image.LimeImageExplainer()

            # Define prediction function
            def predict_fn(images):
                results = []
                for img in images:
                    enhanced = enhancement_fn(img)
                    # Use difference as score
                    diff = np.abs(enhanced.astype(float) - img.astype(float)).mean()
                    results.append([1 - diff / 255, diff / 255])
                return np.array(results)

            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate explanation
            explanation = explainer.explain_instance(
                image_rgb,
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=num_samples,
            )

            # Get important superpixels
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=num_features,
                hide_rest=False,
            )

            # Create visualization
            visualization = mark_boundaries(temp / 255.0, mask)
            visualization = (visualization * 255).astype(np.uint8)
            visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

            metadata = {
                "method": "lime",
                "num_samples": num_samples,
                "num_features": num_features,
            }

            return visualization, metadata

        except ImportError:
            logger.error("LIME not installed. Install with: pip install lime")
            return image, {"error": "LIME not available"}


class SHAP:
    """
    SHapley Additive exPlanations.

    Uses game theory to explain model predictions.
    """

    def __init__(self):
        """Initialize SHAP."""
        logger.info("SHAP initialized")

    def explain_enhancement(
        self,
        image: np.ndarray,
        model: nn.Module,
        background_samples: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Explain enhancement using SHAP.

        Args:
            image: Input image
            model: Enhancement model
            background_samples: Background dataset for SHAP

        Returns:
            Tuple of (shap_values_visualization, metadata)
        """
        try:
            import shap

            # Create explainer
            if background_samples is not None:
                # Convert to tensors
                background = torch.stack([
                    self._prepare_image(img) for img in background_samples
                ])
                explainer = shap.DeepExplainer(model, background)
            else:
                # Use gradient explainer
                explainer = shap.GradientExplainer(model, image)

            # Calculate SHAP values
            img_tensor = self._prepare_image(image)
            shap_values = explainer.shap_values(img_tensor)

            # Visualize
            # Convert SHAP values to heatmap
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_abs = np.abs(shap_values).sum(axis=1)  # Sum across channels
            shap_abs = (shap_abs - shap_abs.min()) / (shap_abs.max() - shap_abs.min())

            # Resize to image size
            h, w = image.shape[:2]
            shap_heatmap = cv2.resize(shap_abs.squeeze(), (w, h))
            shap_heatmap = (shap_heatmap * 255).astype(np.uint8)

            # Apply colormap
            visualization = cv2.applyColorMap(shap_heatmap, cv2.COLORMAP_JET)

            metadata = {
                "method": "shap",
                "has_background": background_samples is not None,
            }

            return visualization, metadata

        except ImportError:
            logger.error("SHAP not installed. Install with: pip install shap")
            return image, {"error": "SHAP not available"}

    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image for model."""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        return torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)


class ActivationMaximization:
    """
    Activation maximization to visualize what features the model detects.

    Generates images that maximally activate specific neurons/filters.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize activation maximization.

        Args:
            model: PyTorch model
        """
        self.model = model
        logger.info("Activation maximization initialized")

    def visualize_filter(
        self,
        layer_name: str,
        filter_idx: int,
        image_size: Tuple[int, int] = (224, 224),
        num_iterations: int = 100,
    ) -> np.ndarray:
        """
        Visualize what a filter detects.

        Args:
            layer_name: Name of layer
            filter_idx: Filter index
            image_size: Size of generated image
            num_iterations: Optimization iterations

        Returns:
            Visualization image
        """
        h, w = image_size

        # Create random input
        input_img = torch.randn(1, 3, h, w, requires_grad=True)

        if next(self.model.parameters()).is_cuda:
            input_img = input_img.cuda()

        # Optimizer
        optimizer = torch.optim.Adam([input_img], lr=0.1)

        # Get target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_layer = module
                break

        if target_layer is None:
            logger.error(f"Layer {layer_name} not found")
            return np.zeros((h, w, 3), dtype=np.uint8)

        # Hook to capture activation
        activation = {}

        def hook(module, input, output):
            activation['output'] = output

        handle = target_layer.register_forward_hook(hook)

        # Optimization loop
        for i in range(num_iterations):
            optimizer.zero_grad()

            # Forward
            self.model(input_img)

            # Get activation
            act = activation['output']

            # Loss: maximize activation of target filter
            if len(act.shape) == 4:  # Conv layer (B, C, H, W)
                loss = -act[0, filter_idx].mean()
            else:  # FC layer
                loss = -act[0, filter_idx]

            # Regularization (total variation)
            tv_loss = torch.sum(torch.abs(input_img[:, :, :, :-1] - input_img[:, :, :, 1:])) + \
                     torch.sum(torch.abs(input_img[:, :, :-1, :] - input_img[:, :, 1:, :]))

            total_loss = loss + 0.01 * tv_loss

            # Backward
            total_loss.backward()
            optimizer.step()

            # Normalize
            with torch.no_grad():
                input_img.data = torch.clamp(input_img.data, -3, 3)

        # Remove hook
        handle.remove()

        # Convert to image
        img_np = input_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = (img_np * 255).astype(np.uint8)

        return img_np


class AttributionMaps:
    """
    Generate attribution maps showing which input pixels affect output.

    Uses various attribution methods: Integrated Gradients, SmoothGrad, etc.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize attribution maps.

        Args:
            model: PyTorch model
        """
        self.model = model
        logger.info("Attribution maps initialized")

    def integrated_gradients(
        self,
        image: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        num_steps: int = 50,
    ) -> np.ndarray:
        """
        Compute integrated gradients.

        Args:
            image: Input image
            baseline: Baseline image (if None, uses black)
            num_steps: Number of integration steps

        Returns:
            Attribution map
        """
        # Prepare inputs
        img_tensor = self._prepare_image(image)

        if baseline is None:
            baseline_tensor = torch.zeros_like(img_tensor)
        else:
            baseline_tensor = self._prepare_image(baseline)

        # Interpolate between baseline and image
        alphas = torch.linspace(0, 1, num_steps).to(img_tensor.device)

        gradients = []

        for alpha in alphas:
            interpolated = baseline_tensor + alpha * (img_tensor - baseline_tensor)
            interpolated.requires_grad = True

            # Forward
            output = self.model(interpolated)

            # Backward
            self.model.zero_grad()
            output.mean().backward()

            # Store gradient
            gradients.append(interpolated.grad.detach())

        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Integrated gradients
        integrated_grads = (img_tensor - baseline_tensor) * avg_gradients

        # Convert to visualization
        ig_map = integrated_grads.squeeze().cpu().numpy()
        ig_map = np.abs(ig_map).sum(axis=0)  # Sum across channels

        # Normalize
        ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min() + 1e-8)
        ig_map = (ig_map * 255).astype(np.uint8)

        return ig_map

    def smooth_grad(
        self,
        image: np.ndarray,
        num_samples: int = 50,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """
        Compute SmoothGrad.

        Args:
            image: Input image
            num_samples: Number of noisy samples
            noise_level: Std of Gaussian noise

        Returns:
            Attribution map
        """
        img_tensor = self._prepare_image(image)

        gradients = []

        for _ in range(num_samples):
            # Add noise
            noise = torch.randn_like(img_tensor) * noise_level
            noisy_img = img_tensor + noise
            noisy_img.requires_grad = True

            # Forward
            output = self.model(noisy_img)

            # Backward
            self.model.zero_grad()
            output.mean().backward()

            # Store
            gradients.append(noisy_img.grad.detach())

        # Average
        smooth_grads = torch.stack(gradients).mean(dim=0)

        # Convert to visualization
        sg_map = smooth_grads.squeeze().cpu().numpy()
        sg_map = np.abs(sg_map).sum(axis=0)

        # Normalize
        sg_map = (sg_map - sg_map.min()) / (sg_map.max() - sg_map.min() + 1e-8)
        sg_map = (sg_map * 255).astype(np.uint8)

        return sg_map

    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image for model."""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

        if next(self.model.parameters()).is_cuda:
            img_tensor = img_tensor.cuda()

        return img_tensor


class ExplainabilityDashboard:
    """
    Comprehensive explainability dashboard.

    Combines multiple XAI methods for complete understanding.
    """

    def __init__(self, model: nn.Module, target_layer: str):
        """
        Initialize dashboard.

        Args:
            model: Model to explain
            target_layer: Target layer for Grad-CAM
        """
        self.model = model
        self.gradcam = GradCAM(model, target_layer)
        self.attribution = AttributionMaps(model)
        self.activation_max = ActivationMaximization(model)

        logger.info("Explainability dashboard initialized")

    def generate_full_explanation(
        self,
        image: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete explanation with all methods.

        Args:
            image: Input image

        Returns:
            Dictionary of explanation visualizations
        """
        explanations = {}

        # Grad-CAM
        cam = self.gradcam.generate_cam(image)
        explanations["gradcam"] = self.gradcam.visualize_cam(image, cam)

        # Integrated Gradients
        ig_map = self.attribution.integrated_gradients(image)
        ig_colored = cv2.applyColorMap(ig_map, cv2.COLORMAP_JET)
        explanations["integrated_gradients"] = ig_colored

        # SmoothGrad
        sg_map = self.attribution.smooth_grad(image)
        sg_colored = cv2.applyColorMap(sg_map, cv2.COLORMAP_JET)
        explanations["smoothgrad"] = sg_colored

        return explanations
