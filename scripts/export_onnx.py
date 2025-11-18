#!/usr/bin/env python3
"""Export models to ONNX format for optimized deployment."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from loguru import logger

from face_enhancement.models.optimization import ModelOptimizer, ONNXInference


def main():
    """Export models to ONNX."""
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to PyTorch model")
    parser.add_argument("--output", type=Path, required=True, help="Output ONNX file path")
    parser.add_argument("--input-shape", nargs=4, type=int, default=[1, 3, 512, 512],
                       help="Input shape (batch, channels, height, width)")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark exported model")
    args = parser.parse_args()

    logger.info("Exporting model to ONNX...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Input shape: {args.input_shape}")

    # Load model
    try:
        model = torch.load(args.model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Export to ONNX
    optimizer = ModelOptimizer()
    success = optimizer.export_to_onnx(
        model,
        args.output,
        input_shape=tuple(args.input_shape),
        opset_version=args.opset,
    )

    if not success:
        logger.error("Export failed")
        return

    logger.info("✓ Export successful!")

    # Benchmark if requested
    if args.benchmark:
        logger.info("\nBenchmarking ONNX model...")

        try:
            onnx_model = ONNXInference(args.output)
            results = onnx_model.benchmark(
                input_shape=tuple(args.input_shape),
                num_runs=100,
            )

            logger.info("Benchmark results:")
            logger.info(f"  Average time: {results['avg_time']:.4f}s")
            logger.info(f"  Min time: {results['min_time']:.4f}s")
            logger.info(f"  Max time: {results['max_time']:.4f}s")
            logger.info(f"  Throughput: {results['throughput']:.2f} FPS")

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
