"""Visualization utilities for model explainability."""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from pathlib import Path


class EnhancementVisualizer:
    """
    Visualization tools for understanding enhancement process.

    Features:
    - Attention map visualization
    - Before/after comparison grids
    - Quality metric visualization
    - Progressive stage visualization
    - Feature map visualization
    """

    @staticmethod
    def create_comparison_grid(
        images: List[Tuple[str, np.ndarray]],
        grid_size: Optional[Tuple[int, int]] = None,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Create a grid comparison of multiple images.

        Args:
            images: List of (label, image) tuples
            grid_size: Optional (rows, cols) for grid layout
            output_path: Optional path to save visualization

        Returns:
            Grid image
        """
        n_images = len(images)

        if grid_size is None:
            # Auto-calculate grid size
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
        else:
            rows, cols = grid_size

        # Get max dimensions
        max_h = max(img.shape[0] for _, img in images)
        max_w = max(img.shape[1] for _, img in images)

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)

        # Plot images
        for idx, (label, img) in enumerate(images):
            row = idx // cols
            col = idx % cols

            # Convert BGR to RGB for display
            if len(img.shape) == 3 and img.shape[2] == 3:
                display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                display_img = img

            axes[row, col].imshow(display_img)
            axes[row, col].set_title(label, fontsize=12, fontweight='bold')
            axes[row, col].axis('off')

        # Remove empty subplots
        for idx in range(n_images, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

        # Convert to numpy array
        fig.canvas.draw()
        grid_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        grid_img = grid_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return grid_img

    @staticmethod
    def visualize_progressive_stages(
        stage_outputs: List[Tuple[str, np.ndarray]],
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Visualize progressive enhancement stages.

        Args:
            stage_outputs: List of (stage_name, image) tuples
            output_path: Optional path to save

        Returns:
            Visualization image
        """
        return EnhancementVisualizer.create_comparison_grid(
            stage_outputs,
            grid_size=(1, len(stage_outputs)),
            output_path=output_path,
        )

    @staticmethod
    def create_attention_map(
        image: np.ndarray,
        attention_weights: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Create attention map visualization.

        Args:
            image: Original image
            attention_weights: Attention weight map (0-1)
            colormap: OpenCV colormap

        Returns:
            Attention visualization
        """
        # Normalize attention weights
        attention_norm = (attention_weights * 255).astype(np.uint8)

        # Resize to match image
        if attention_norm.shape[:2] != image.shape[:2]:
            attention_norm = cv2.resize(
                attention_norm, (image.shape[1], image.shape[0])
            )

        # Apply colormap
        attention_colored = cv2.applyColorMap(attention_norm, colormap)

        # Blend with original image
        blended = cv2.addWeighted(image, 0.6, attention_colored, 0.4, 0)

        return blended

    @staticmethod
    def plot_quality_metrics(
        metrics: dict,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Plot quality metrics as bar chart.

        Args:
            metrics: Dictionary of metric names to values
            output_path: Optional path to save

        Returns:
            Plot image
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        # Create bar chart
        bars = ax.bar(metric_names, metric_values, color='skyblue', edgecolor='navy')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontweight='bold',
            )

        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Image Quality Metrics', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        # Convert to numpy array
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return plot_img

    @staticmethod
    def plot_metrics_comparison(
        original_metrics: dict,
        enhanced_metrics: dict,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Compare metrics between original and enhanced images.

        Args:
            original_metrics: Metrics for original image
            enhanced_metrics: Metrics for enhanced image
            output_path: Optional path to save

        Returns:
            Comparison plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Common metrics
        metrics = set(original_metrics.keys()) & set(enhanced_metrics.keys())
        metrics = sorted(list(metrics))

        x = np.arange(len(metrics))
        width = 0.35

        orig_values = [original_metrics[m] for m in metrics]
        enh_values = [enhanced_metrics[m] for m in metrics]

        bars1 = ax.bar(x - width/2, orig_values, width, label='Original', color='coral')
        bars2 = ax.bar(x + width/2, enh_values, width, label='Enhanced', color='lightgreen')

        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Quality Metrics: Original vs Enhanced', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')

        # Convert to numpy array
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return plot_img

    @staticmethod
    def create_side_by_side(
        original: np.ndarray,
        enhanced: np.ndarray,
        labels: Tuple[str, str] = ("Original", "Enhanced"),
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Create side-by-side comparison.

        Args:
            original: Original image
            enhanced: Enhanced image
            labels: Tuple of (original_label, enhanced_label)
            output_path: Optional save path

        Returns:
            Side-by-side comparison image
        """
        return EnhancementVisualizer.create_comparison_grid(
            [(labels[0], original), (labels[1], enhanced)],
            grid_size=(1, 2),
            output_path=output_path,
        )

    @staticmethod
    def create_detail_zoom(
        original: np.ndarray,
        enhanced: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        zoom_factor: int = 2,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Create detail zoom comparison of a region.

        Args:
            original: Original image
            enhanced: Enhanced image
            roi: Region of interest (x, y, w, h), or None for center
            zoom_factor: Zoom magnification
            output_path: Optional save path

        Returns:
            Detail zoom visualization
        """
        h, w = original.shape[:2]

        if roi is None:
            # Default to center region
            roi_w = w // 4
            roi_h = h // 4
            roi = (w // 2 - roi_w // 2, h // 2 - roi_h // 2, roi_w, roi_h)

        x, y, roi_w, roi_h = roi

        # Extract ROIs
        orig_roi = original[y:y+roi_h, x:x+roi_w]
        enh_roi = enhanced[y:y+roi_h, x:x+roi_w]

        # Zoom
        orig_zoom = cv2.resize(
            orig_roi,
            (roi_w * zoom_factor, roi_h * zoom_factor),
            interpolation=cv2.INTER_NEAREST,
        )
        enh_zoom = cv2.resize(
            enh_roi,
            (roi_w * zoom_factor, roi_h * zoom_factor),
            interpolation=cv2.INTER_NEAREST,
        )

        # Draw ROI boxes on full images
        orig_with_box = original.copy()
        enh_with_box = enhanced.copy()
        cv2.rectangle(orig_with_box, (x, y), (x+roi_w, y+roi_h), (0, 255, 0), 2)
        cv2.rectangle(enh_with_box, (x, y), (x+roi_w, y+roi_h), (0, 255, 0), 2)

        # Create grid
        images = [
            ("Original", orig_with_box),
            ("Original (Zoom)", orig_zoom),
            ("Enhanced", enh_with_box),
            ("Enhanced (Zoom)", enh_zoom),
        ]

        return EnhancementVisualizer.create_comparison_grid(
            images,
            grid_size=(2, 2),
            output_path=output_path,
        )

    @staticmethod
    def create_difference_map(
        original: np.ndarray,
        enhanced: np.ndarray,
        amplify: float = 3.0,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Create difference map showing changes.

        Args:
            original: Original image
            enhanced: Enhanced image
            amplify: Amplification factor for differences
            output_path: Optional save path

        Returns:
            Difference map visualization
        """
        # Ensure same size
        if original.shape != enhanced.shape:
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))

        # Calculate difference
        diff = cv2.absdiff(enhanced, original)

        # Amplify
        diff_amp = np.clip(diff * amplify, 0, 255).astype(np.uint8)

        # Apply colormap
        diff_colored = cv2.applyColorMap(
            cv2.cvtColor(diff_amp, cv2.COLOR_BGR2GRAY),
            cv2.COLORMAP_JET,
        )

        # Create visualization
        images = [
            ("Original", original),
            ("Enhanced", enhanced),
            ("Difference (amplified)", diff_colored),
        ]

        return EnhancementVisualizer.create_comparison_grid(
            images,
            grid_size=(1, 3),
            output_path=output_path,
        )
