"""Interactive Gradio Web UI for Face Enhancement.

Provides a modern web interface for all enhancement features.
Supports single image, batch, real-time, and progressive enhancement.
"""

from typing import Optional, Tuple, List
import gradio as gr
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from face_enhancement.config import AppConfig
from face_enhancement.core.pipeline import EnhancementPipeline
from face_enhancement.models.progressive_enhancer import ProgressiveEnhancer
from face_enhancement.models.ensemble import ModelEnsemble
from face_enhancement.utils.quality_metrics import QualityAssessment
from face_enhancement.utils.visualization import EnhancementVisualizer
from loguru import logger


class GradioFaceEnhancementUI:
    """Gradio-based interactive UI for face enhancement."""

    def __init__(self):
        """Initialize the UI."""
        self.config = AppConfig()
        self.pipeline = None
        self.progressive_enhancer = None
        self.ensemble = None
        self.quality_assessor = QualityAssessment()
        self.visualizer = EnhancementVisualizer()

        logger.info("Gradio UI initialized")

    def _lazy_load_pipeline(self, model: str = "gfpgan"):
        """Lazy load enhancement pipeline."""
        if self.pipeline is None or self.config.enhancement.model != model:
            self.config.enhancement.model = model
            self.pipeline = EnhancementPipeline(self.config)
            logger.info(f"Loaded pipeline with {model}")

    def _lazy_load_progressive(self, model: str = "gfpgan", stages: int = 3):
        """Lazy load progressive enhancer."""
        if self.progressive_enhancer is None:
            self.progressive_enhancer = ProgressiveEnhancer(
                base_model=model,
                stages=stages,
                device=self.config.enhancement.device,
            )
            logger.info(f"Loaded progressive enhancer with {stages} stages")

    def enhance_image(
        self,
        image: np.ndarray,
        model: str,
        upscale_factor: int,
        fidelity_weight: float,
        enable_preprocessing: bool,
        show_quality_metrics: bool,
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Enhance a single image.

        Args:
            image: Input image
            model: Model to use
            upscale_factor: Upscale factor
            fidelity_weight: Fidelity weight
            enable_preprocessing: Enable preprocessing
            show_quality_metrics: Show quality metrics

        Returns:
            Tuple of (enhanced_image, metrics_text)
        """
        if image is None:
            return None, "Please upload an image"

        try:
            # Update config
            self.config.enhancement.model = model
            self.config.enhancement.upscale_factor = upscale_factor
            self.config.enhancement.fidelity_weight = fidelity_weight

            # Load pipeline
            self._lazy_load_pipeline(model)

            # Process
            enhanced, metadata = self.pipeline.process_image(
                image, preprocess=enable_preprocessing
            )

            # Calculate metrics if requested
            metrics_text = None
            if show_quality_metrics:
                metrics = self.quality_assessor.calculate_all_metrics(
                    enhanced, reference=image
                )
                metrics_text = self._format_metrics(metrics, metadata)

            return enhanced, metrics_text

        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return None, f"Error: {str(e)}"

    def progressive_enhance(
        self,
        image: np.ndarray,
        model: str,
        num_stages: int,
        show_stages: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[str]]:
        """
        Progressive multi-stage enhancement.

        Args:
            image: Input image
            model: Base model
            num_stages: Number of stages
            show_stages: Show intermediate stages

        Returns:
            Tuple of (final_enhanced, stage_visualization, info_text)
        """
        if image is None:
            return None, None, "Please upload an image"

        try:
            self._lazy_load_progressive(model, num_stages)

            # Enhance
            enhanced, metadata = self.progressive_enhancer.enhance(
                image, intermediate_outputs=show_stages
            )

            # Create stage visualization
            stage_viz = None
            if show_stages and "stage_outputs" in metadata:
                stage_viz = self.visualizer.visualize_progressive_stages(
                    metadata["stage_outputs"]
                )

            # Info text
            stage_info = self.progressive_enhancer.get_stage_info()
            info_text = f"""**Progressive Enhancement Complete**

Stages: {stage_info['num_stages']}
Upscale factors: {stage_info['stage_upscales']}
Fidelity weights: {[f'{w:.2f}' for w in stage_info['stage_fidelity']]}
"""

            return enhanced, stage_viz, info_text

        except Exception as e:
            logger.error(f"Progressive enhancement failed: {e}")
            return None, None, f"Error: {str(e)}"

    def compare_models(
        self, image: np.ndarray, models: List[str]
    ) -> Tuple[List[np.ndarray], str]:
        """
        Compare multiple models on the same image.

        Args:
            image: Input image
            models: List of model names

        Returns:
            Tuple of (enhanced_images, comparison_text)
        """
        if image is None:
            return [], "Please upload an image"

        results = []
        metrics_list = []

        try:
            for model in models:
                self.config.enhancement.model = model
                self._lazy_load_pipeline(model)

                enhanced, _ = self.pipeline.process_image(image)
                results.append(enhanced)

                # Calculate metrics
                metrics = self.quality_assessor.calculate_all_metrics(
                    enhanced, reference=image
                )
                metrics["model"] = model
                metrics_list.append(metrics)

            # Format comparison
            comparison_text = "**Model Comparison**\n\n"
            for m in metrics_list:
                comparison_text += f"**{m['model'].upper()}**\n"
                comparison_text += f"- Sharpness: {m['sharpness']:.2f}\n"
                comparison_text += f"- Contrast: {m['contrast']:.2f}\n"
                comparison_text += f"- Colorfulness: {m['colorfulness']:.2f}\n"
                if m.get("psnr"):
                    comparison_text += f"- PSNR: {m['psnr']:.2f}\n"
                comparison_text += "\n"

            return results, comparison_text

        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return [], f"Error: {str(e)}"

    def batch_enhance(
        self, files: List[str], model: str, upscale_factor: int
    ) -> Tuple[List[np.ndarray], str]:
        """
        Batch process multiple images.

        Args:
            files: List of file paths
            model: Model to use
            upscale_factor: Upscale factor

        Returns:
            Tuple of (enhanced_images, summary_text)
        """
        if not files:
            return [], "Please upload images"

        results = []
        summary = []

        try:
            self.config.enhancement.model = model
            self.config.enhancement.upscale_factor = upscale_factor
            self._lazy_load_pipeline(model)

            for file_path in files:
                image = cv2.imread(file_path)
                if image is None:
                    summary.append(f"❌ Failed to load: {Path(file_path).name}")
                    continue

                enhanced, _ = self.pipeline.process_image(image)
                results.append(enhanced)
                summary.append(f"✓ Enhanced: {Path(file_path).name}")

            summary_text = "\n".join(summary)
            return results, summary_text

        except Exception as e:
            logger.error(f"Batch enhancement failed: {e}")
            return [], f"Error: {str(e)}"

    def _format_metrics(self, metrics: dict, metadata: dict) -> str:
        """Format metrics for display."""
        text = "**Quality Metrics**\n\n"

        text += f"**Enhancement Info**\n"
        text += f"- Model: {metadata.get('model', 'N/A')}\n"
        text += f"- Faces detected: {metadata.get('num_faces', 0)}\n"
        text += f"- Processing time: {metadata.get('processing_time', 0):.2f}s\n\n"

        text += "**Image Quality**\n"
        text += f"- Sharpness: {metrics.get('sharpness', 0):.2f}\n"
        text += f"- Brightness: {metrics.get('brightness', 0):.2f}\n"
        text += f"- Contrast: {metrics.get('contrast', 0):.2f}\n"
        text += f"- Colorfulness: {metrics.get('colorfulness', 0):.2f}\n"
        text += f"- Entropy: {metrics.get('entropy', 0):.2f}\n"

        if metrics.get("psnr"):
            text += f"\n**Reference Metrics**\n"
            text += f"- PSNR: {metrics.get('psnr', 0):.2f} dB\n"
            text += f"- SSIM: {metrics.get('ssim', 0):.4f}\n"

        return text

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(
            title="Face Enhancement Studio",
            theme=gr.themes.Soft(),
        ) as demo:
            gr.Markdown(
                """
            # 🎨 Face Enhancement Studio

            State-of-the-art face enhancement using deep learning.
            Choose from multiple enhancement modes below.
            """
            )

            with gr.Tabs():
                # Tab 1: Simple Enhancement
                with gr.Tab("Simple Enhancement"):
                    with gr.Row():
                        with gr.Column():
                            simple_input = gr.Image(label="Upload Image")
                            simple_model = gr.Dropdown(
                                choices=["gfpgan", "codeformer"],
                                value="gfpgan",
                                label="Model",
                            )
                            simple_upscale = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=2,
                                step=1,
                                label="Upscale Factor",
                            )
                            simple_fidelity = gr.Slider(
                                minimum=0,
                                maximum=1,
                                value=0.5,
                                step=0.1,
                                label="Fidelity Weight",
                            )
                            simple_preprocess = gr.Checkbox(
                                value=True, label="Enable Preprocessing"
                            )
                            simple_show_metrics = gr.Checkbox(
                                value=True, label="Show Quality Metrics"
                            )
                            simple_btn = gr.Button("Enhance", variant="primary")

                        with gr.Column():
                            simple_output = gr.Image(label="Enhanced Image")
                            simple_metrics = gr.Markdown(label="Metrics")

                    simple_btn.click(
                        fn=self.enhance_image,
                        inputs=[
                            simple_input,
                            simple_model,
                            simple_upscale,
                            simple_fidelity,
                            simple_preprocess,
                            simple_show_metrics,
                        ],
                        outputs=[simple_output, simple_metrics],
                    )

                # Tab 2: Progressive Enhancement
                with gr.Tab("Progressive Enhancement"):
                    with gr.Row():
                        with gr.Column():
                            prog_input = gr.Image(label="Upload Image")
                            prog_model = gr.Dropdown(
                                choices=["gfpgan", "codeformer"],
                                value="gfpgan",
                                label="Base Model",
                            )
                            prog_stages = gr.Slider(
                                minimum=2,
                                maximum=4,
                                value=3,
                                step=1,
                                label="Number of Stages",
                            )
                            prog_show_stages = gr.Checkbox(
                                value=True, label="Show Intermediate Stages"
                            )
                            prog_btn = gr.Button("Progressive Enhance", variant="primary")

                        with gr.Column():
                            prog_output = gr.Image(label="Final Enhanced")
                            prog_stages_viz = gr.Image(label="Stage Visualization")
                            prog_info = gr.Markdown(label="Enhancement Info")

                    prog_btn.click(
                        fn=self.progressive_enhance,
                        inputs=[prog_input, prog_model, prog_stages, prog_show_stages],
                        outputs=[prog_output, prog_stages_viz, prog_info],
                    )

                # Tab 3: Model Comparison
                with gr.Tab("Model Comparison"):
                    with gr.Row():
                        with gr.Column():
                            comp_input = gr.Image(label="Upload Image")
                            comp_models = gr.CheckboxGroup(
                                choices=["gfpgan", "codeformer"],
                                value=["gfpgan", "codeformer"],
                                label="Models to Compare",
                            )
                            comp_btn = gr.Button("Compare Models", variant="primary")

                        with gr.Column():
                            comp_outputs = gr.Gallery(
                                label="Model Outputs", columns=2, height="auto"
                            )
                            comp_metrics = gr.Markdown(label="Comparison Metrics")

                    comp_btn.click(
                        fn=self.compare_models,
                        inputs=[comp_input, comp_models],
                        outputs=[comp_outputs, comp_metrics],
                    )

                # Tab 4: Batch Processing
                with gr.Tab("Batch Processing"):
                    with gr.Row():
                        with gr.Column():
                            batch_input = gr.File(
                                file_count="multiple", label="Upload Images"
                            )
                            batch_model = gr.Dropdown(
                                choices=["gfpgan", "codeformer"],
                                value="gfpgan",
                                label="Model",
                            )
                            batch_upscale = gr.Slider(
                                minimum=1, maximum=4, value=2, step=1, label="Upscale Factor"
                            )
                            batch_btn = gr.Button("Process Batch", variant="primary")

                        with gr.Column():
                            batch_output = gr.Gallery(
                                label="Enhanced Images", columns=3, height="auto"
                            )
                            batch_summary = gr.Markdown(label="Processing Summary")

                    batch_btn.click(
                        fn=self.batch_enhance,
                        inputs=[batch_input, batch_model, batch_upscale],
                        outputs=[batch_output, batch_summary],
                    )

            gr.Markdown(
                """
            ---

            **Models:**
            - **GFPGAN**: Generative Facial Prior GAN - Best for general face enhancement
            - **CodeFormer**: Transformer-based - Better for heavily degraded images

            **Features:**
            - Single image enhancement with quality metrics
            - Progressive multi-stage enhancement
            - Model comparison and benchmarking
            - Batch processing for multiple images

            **Tips:**
            - Higher upscale factors = better resolution but slower processing
            - Higher fidelity weight = preserve original features more
            - Enable preprocessing for low-light or noisy images
            """
            )

        return demo


def launch_ui(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
):
    """
    Launch the Gradio UI.

    Args:
        server_name: Server host
        server_port: Server port
        share: Create public share link
    """
    logger.info(f"Launching Gradio UI on {server_name}:{server_port}")

    ui = GradioFaceEnhancementUI()
    demo = ui.create_interface()

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Face Enhancement Web UI")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create public share link"
    )

    args = parser.parse_args()

    launch_ui(server_name=args.host, server_port=args.port, share=args.share)
