"""Model optimization utilities for deployment."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class ModelOptimizer:
    """
    Optimize models for production deployment.

    Supports:
    - ONNX export
    - TorchScript export
    - Quantization
    - Pruning
    """

    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        output_path: Path,
        input_shape: Tuple[int, ...] = (1, 3, 512, 512),
        opset_version: int = 14,
        dynamic_axes: Optional[dict] = None,
    ) -> bool:
        """
        Export PyTorch model to ONNX format.

        Args:
            model: PyTorch model
            output_path: Output ONNX file path
            input_shape: Input tensor shape
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes configuration

        Returns:
            Success status
        """
        try:
            model.eval()

            # Create dummy input
            dummy_input = torch.randn(*input_shape)

            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                }

            # Export
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
            )

            logger.info(f"Model exported to ONNX: {output_path}")
            return True

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False

    @staticmethod
    def export_to_torchscript(
        model: nn.Module,
        output_path: Path,
        input_shape: Tuple[int, ...] = (1, 3, 512, 512),
        method: str = "trace",
    ) -> bool:
        """
        Export PyTorch model to TorchScript.

        Args:
            model: PyTorch model
            output_path: Output file path
            input_shape: Input tensor shape
            method: Export method ('trace' or 'script')

        Returns:
            Success status
        """
        try:
            model.eval()

            if method == "trace":
                dummy_input = torch.randn(*input_shape)
                traced_model = torch.jit.trace(model, dummy_input)
                traced_model.save(str(output_path))

            elif method == "script":
                scripted_model = torch.jit.script(model)
                scripted_model.save(str(output_path))

            else:
                raise ValueError(f"Unknown method: {method}")

            logger.info(f"Model exported to TorchScript: {output_path}")
            return True

        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            return False

    @staticmethod
    def quantize_model(
        model: nn.Module,
        output_path: Optional[Path] = None,
        quantization_type: str = "dynamic",
    ) -> nn.Module:
        """
        Quantize model for smaller size and faster inference.

        Args:
            model: PyTorch model
            output_path: Optional output path
            quantization_type: 'dynamic' or 'static'

        Returns:
            Quantized model
        """
        try:
            model.eval()

            if quantization_type == "dynamic":
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear, nn.Conv2d},
                    dtype=torch.qint8
                )

            elif quantization_type == "static":
                # Prepare for static quantization
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                quantized_model = torch.quantization.prepare(model, inplace=False)
                # Note: Need calibration data for static quantization
                quantized_model = torch.quantization.convert(quantized_model, inplace=False)

            else:
                raise ValueError(f"Unknown quantization type: {quantization_type}")

            if output_path:
                torch.save(quantized_model.state_dict(), output_path)
                logger.info(f"Quantized model saved to: {output_path}")

            logger.info(f"Model quantized using {quantization_type} quantization")
            return quantized_model

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model

    @staticmethod
    def optimize_for_inference(
        model: nn.Module,
        device: str = "cuda",
    ) -> nn.Module:
        """
        Optimize model for inference.

        Args:
            model: PyTorch model
            device: Target device

        Returns:
            Optimized model
        """
        model.eval()

        # Move to device
        model = model.to(device)

        # Enable inference mode optimizations
        if device == "cuda":
            # Use mixed precision
            model = model.half()  # FP16

        # Fuse operations
        try:
            model = torch.jit.optimize_for_inference(torch.jit.script(model))
            logger.info("Model optimized for inference")
        except Exception as e:
            logger.warning(f"JIT optimization failed: {e}")

        return model

    @staticmethod
    def benchmark_model(
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 512, 512),
        num_runs: int = 100,
        warmup_runs: int = 10,
        device: str = "cuda",
    ) -> dict:
        """
        Benchmark model performance.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            device: Device to run on

        Returns:
            Benchmark results
        """
        import time

        model.eval()
        model = model.to(device)

        dummy_input = torch.randn(*input_shape).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)

        # Benchmark
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(dummy_input)

                if device == "cuda":
                    torch.cuda.synchronize()

                elapsed = time.time() - start
                times.append(elapsed)

        times_arr = np.array(times)

        results = {
            "avg_time": float(np.mean(times_arr)),
            "min_time": float(np.min(times_arr)),
            "max_time": float(np.max(times_arr)),
            "std_time": float(np.std(times_arr)),
            "throughput": float(num_runs / np.sum(times_arr)),
            "input_shape": input_shape,
            "device": device,
        }

        logger.info(f"Benchmark: avg={results['avg_time']:.4f}s, throughput={results['throughput']:.2f} FPS")

        return results


class ONNXInference:
    """ONNX inference engine."""

    def __init__(self, model_path: Path, providers: list = None):
        """
        Initialize ONNX inference.

        Args:
            model_path: Path to ONNX model
            providers: Execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        try:
            import onnxruntime as ort
            self.ort = ort
        except ImportError:
            raise ImportError("onnxruntime not available. Install with: pip install onnxruntime")

        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"ONNX model loaded: {model_path}")
        logger.info(f"Providers: {self.session.get_providers()}")

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference.

        Args:
            input_data: Input numpy array

        Returns:
            Output numpy array
        """
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        return outputs[0]

    def benchmark(self, input_shape: Tuple[int, ...], num_runs: int = 100) -> dict:
        """Benchmark ONNX inference."""
        import time

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            _ = self.infer(dummy_input)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.infer(dummy_input)
            elapsed = time.time() - start
            times.append(elapsed)

        times_arr = np.array(times)

        return {
            "avg_time": float(np.mean(times_arr)),
            "min_time": float(np.min(times_arr)),
            "max_time": float(np.max(times_arr)),
            "throughput": float(num_runs / np.sum(times_arr)),
        }
