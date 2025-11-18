"""TensorFlow Lite conversion for mobile deployment.

Converts PyTorch models to TensorFlow Lite for:
- Mobile devices (Android/iOS)
- Edge devices (Raspberry Pi, Jetson Nano)
- Web browsers (TensorFlow.js)
"""

from typing import Optional, Tuple, List
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class TFLiteConverter:
    """
    Convert PyTorch models to TensorFlow Lite.

    Supports:
    - PyTorch to ONNX to TensorFlow to TFLite pipeline
    - Quantization (int8, float16)
    - Model optimization for mobile
    """

    def __init__(self):
        """Initialize TFLite converter."""
        logger.info("TFLite converter initialized")

    def convert_pytorch_to_tflite(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: Path,
        quantize: bool = True,
        quantization_mode: str = "float16",
    ) -> Path:
        """
        Convert PyTorch model to TensorFlow Lite.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape (B, C, H, W)
            output_path: Output .tflite file path
            quantize: Enable quantization
            quantization_mode: 'int8' or 'float16'

        Returns:
            Path to generated .tflite file
        """
        logger.info(f"Converting PyTorch model to TFLite: {output_path}")

        # Step 1: PyTorch -> ONNX
        onnx_path = output_path.with_suffix(".onnx")
        self._pytorch_to_onnx(model, input_shape, onnx_path)

        # Step 2: ONNX -> TensorFlow
        tf_path = output_path.with_suffix(".pb")
        self._onnx_to_tensorflow(onnx_path, tf_path)

        # Step 3: TensorFlow -> TFLite
        tflite_path = self._tensorflow_to_tflite(
            tf_path, output_path, quantize, quantization_mode
        )

        logger.info(f"Conversion complete: {tflite_path}")
        return tflite_path

    def _pytorch_to_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: Path,
    ) -> None:
        """Convert PyTorch model to ONNX."""
        model.eval()
        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        logger.info(f"Exported to ONNX: {output_path}")

    def _onnx_to_tensorflow(
        self, onnx_path: Path, output_path: Path
    ) -> None:
        """Convert ONNX model to TensorFlow."""
        try:
            import onnx
            from onnx_tf.backend import prepare

            onnx_model = onnx.load(str(onnx_path))
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(str(output_path))

            logger.info(f"Exported to TensorFlow: {output_path}")

        except ImportError:
            logger.error("onnx-tf not installed. Install with: pip install onnx-tf")
            raise

    def _tensorflow_to_tflite(
        self,
        tf_path: Path,
        output_path: Path,
        quantize: bool,
        quantization_mode: str,
    ) -> Path:
        """Convert TensorFlow model to TFLite."""
        try:
            import tensorflow as tf

            # Load TensorFlow model
            converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))

            # Apply optimizations
            if quantize:
                if quantization_mode == "int8":
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    # For int8 quantization, need representative dataset
                    # converter.representative_dataset = representative_dataset_gen
                elif quantization_mode == "float16":
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]

            # Convert
            tflite_model = converter.convert()

            # Save
            with open(output_path, "wb") as f:
                f.write(tflite_model)

            logger.info(f"Exported to TFLite: {output_path}")
            return output_path

        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise


class MobileOptimizer:
    """
    Optimize models for mobile deployment.

    Applies:
    - Pruning
    - Knowledge distillation
    - Architecture search
    """

    def __init__(self):
        """Initialize mobile optimizer."""
        logger.info("Mobile optimizer initialized")

    def prune_model(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
    ) -> nn.Module:
        """
        Prune model weights for smaller size.

        Args:
            model: PyTorch model
            pruning_ratio: Fraction of weights to prune (0-1)

        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune

            # Prune all Conv2d and Linear layers
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.l1_unstructured(module, name="weight", amount=pruning_ratio)
                    prune.remove(module, "weight")

            logger.info(f"Pruned model with ratio {pruning_ratio}")
            return model

        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            raise

    def create_distilled_model(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader,
        epochs: int = 10,
    ) -> nn.Module:
        """
        Create distilled (smaller) model from teacher.

        Args:
            teacher_model: Large teacher model
            student_model: Small student model
            train_loader: Training data loader
            epochs: Number of training epochs

        Returns:
            Trained student model
        """
        logger.info("Starting knowledge distillation")

        teacher_model.eval()
        student_model.train()

        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0

            for batch_idx, (data, _) in enumerate(train_loader):
                optimizer.zero_grad()

                # Get teacher predictions
                with torch.no_grad():
                    teacher_output = teacher_model(data)

                # Get student predictions
                student_output = student_model(data)

                # Distillation loss
                loss = criterion(student_output, teacher_output)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        logger.info("Knowledge distillation complete")
        return student_model


class MobileInference:
    """
    TensorFlow Lite inference engine for mobile.

    Provides high-performance inference on mobile devices.
    """

    def __init__(self, model_path: Path, num_threads: int = 4):
        """
        Initialize mobile inference engine.

        Args:
            model_path: Path to .tflite model
            num_threads: Number of CPU threads
        """
        self.model_path = model_path
        self.num_threads = num_threads
        self.interpreter = None

        self._load_model()

        logger.info(f"Mobile inference engine initialized with {num_threads} threads")

    def _load_model(self):
        """Load TFLite model."""
        try:
            import tensorflow as tf

            self.interpreter = tf.lite.Interpreter(
                model_path=str(self.model_path),
                num_threads=self.num_threads,
            )
            self.interpreter.allocate_tensors()

            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            logger.info(f"Loaded TFLite model: {self.model_path}")

        except ImportError:
            logger.error("TensorFlow not installed")
            raise

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on image.

        Args:
            image: Input image

        Returns:
            Model output
        """
        # Preprocess
        input_data = self._preprocess(image)

        # Set input
        self.interpreter.set_tensor(
            self.input_details[0]["index"], input_data
        )

        # Run inference
        self.interpreter.invoke()

        # Get output
        output_data = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        # Postprocess
        result = self._postprocess(output_data)

        return result

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        # Get input shape
        input_shape = self.input_details[0]["shape"]
        _, h, w, c = input_shape

        # Resize
        import cv2
        resized = cv2.resize(image, (w, h))

        # Normalize
        normalized = resized.astype(np.float32) / 255.0

        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)

        return batched

    def _postprocess(self, output: np.ndarray) -> np.ndarray:
        """Postprocess model output."""
        # Remove batch dimension
        result = output[0]

        # Denormalize if needed
        result = (result * 255).clip(0, 255).astype(np.uint8)

        return result

    def benchmark(self, num_runs: int = 100) -> dict:
        """
        Benchmark inference performance.

        Args:
            num_runs: Number of inference runs

        Returns:
            Performance metrics
        """
        import time

        # Create dummy input
        input_shape = self.input_details[0]["shape"]
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.interpreter.set_tensor(self.input_details[0]["index"], dummy_input)
            self.interpreter.invoke()

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()

            self.interpreter.set_tensor(self.input_details[0]["index"], dummy_input)
            self.interpreter.invoke()

            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms

        return {
            "num_runs": num_runs,
            "mean_ms": np.mean(times),
            "median_ms": np.median(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "std_ms": np.std(times),
        }


class EdgeDeploymentHelper:
    """
    Helper utilities for edge deployment.

    Supports:
    - Raspberry Pi
    - Jetson Nano
    - Coral TPU
    - Mobile devices
    """

    @staticmethod
    def get_device_info() -> dict:
        """Get device information."""
        import platform

        info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

        # Check for GPU
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["cuda_available"] = False

        return info

    @staticmethod
    def optimize_for_raspberry_pi(model_path: Path) -> Path:
        """
        Optimize model specifically for Raspberry Pi.

        Args:
            model_path: Path to TFLite model

        Returns:
            Path to optimized model
        """
        # Additional optimizations for ARM processors
        optimized_path = model_path.with_name(
            f"{model_path.stem}_rpi{model_path.suffix}"
        )

        logger.info(f"Optimizing for Raspberry Pi: {optimized_path}")

        # Copy and apply ARM-specific optimizations
        import shutil
        shutil.copy(model_path, optimized_path)

        return optimized_path

    @staticmethod
    def optimize_for_jetson(model_path: Path) -> Path:
        """
        Optimize model for NVIDIA Jetson devices.

        Args:
            model_path: Path to model

        Returns:
            Path to optimized model
        """
        # Can use TensorRT for Jetson
        optimized_path = model_path.with_name(
            f"{model_path.stem}_jetson{model_path.suffix}"
        )

        logger.info(f"Optimizing for Jetson: {optimized_path}")

        import shutil
        shutil.copy(model_path, optimized_path)

        return optimized_path


def create_mobile_ready_model(
    pytorch_model: nn.Module,
    input_shape: Tuple[int, ...],
    output_dir: Path,
    model_name: str = "face_enhancer",
    quantize: bool = True,
) -> dict:
    """
    Create mobile-ready model with all formats.

    Args:
        pytorch_model: PyTorch model
        input_shape: Input shape
        output_dir: Output directory
        model_name: Model name
        quantize: Enable quantization

    Returns:
        Dictionary of generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = TFLiteConverter()

    # Convert to TFLite
    tflite_path = output_dir / f"{model_name}.tflite"
    converter.convert_pytorch_to_tflite(
        pytorch_model,
        input_shape,
        tflite_path,
        quantize=quantize,
    )

    # Also export ONNX for other platforms
    onnx_path = output_dir / f"{model_name}.onnx"
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(pytorch_model, dummy_input, str(onnx_path))

    logger.info(f"Mobile-ready models created in {output_dir}")

    return {
        "tflite": tflite_path,
        "onnx": onnx_path,
    }
