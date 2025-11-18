#!/usr/bin/env python3
"""Comprehensive benchmark script for face enhancement pipeline."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from face_enhancement.config import AppConfig
from face_enhancement.core.pipeline import EnhancementPipeline
from face_enhancement.utils.benchmark import PerformanceBenchmark, create_test_images
from loguru import logger


def main():
    """Run comprehensive benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark face enhancement pipeline")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", default="gfpgan", choices=["gfpgan", "codeformer"])
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()

    logger.info("Starting comprehensive benchmark")
    logger.info(f"Device: {args.device}, Model: {args.model}, Runs: {args.runs}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Initialize benchmark
    benchmark = PerformanceBenchmark()

    # Create test images
    logger.info("Creating test images...")
    test_images = create_test_images([
        (480, 640),    # VGA
        (720, 1280),   # HD
        (1080, 1920),  # Full HD
    ])

    # Initialize pipeline
    logger.info("Initializing pipeline...")
    config = AppConfig()
    config.device = args.device
    config.enhancement.model = args.model
    pipeline = EnhancementPipeline(config)

    # Benchmark different configurations
    logger.info("\n" + "="*80)
    logger.info("BENCHMARKING DIFFERENT CONFIGURATIONS")
    logger.info("="*80 + "\n")

    variations = {
        "baseline": {},
        "upscale_2x": {"enhancement.upscale_factor": 2},
        "upscale_4x": {"enhancement.upscale_factor": 4},
        "with_preprocessing": {
            "preprocessing.denoise": True,
            "preprocessing.auto_contrast": True,
            "preprocessing.enhance_brightness": True,
        },
        "no_preprocessing": {
            "preprocessing.denoise": False,
            "preprocessing.auto_contrast": False,
            "preprocessing.enhance_brightness": False,
        },
    }

    results = benchmark.benchmark_pipeline(
        pipeline,
        test_images[:1],  # Use VGA image
        variations=variations
    )

    # Benchmark different image sizes
    logger.info("\n" + "="*80)
    logger.info("BENCHMARKING DIFFERENT IMAGE SIZES")
    logger.info("="*80 + "\n")

    for i, img in enumerate(test_images):
        size_name = f"{img.shape[1]}x{img.shape[0]}"
        logger.info(f"Benchmarking size: {size_name}")

        def process():
            pipeline.process_image(img)

        benchmark.benchmark_function(
            process,
            args=(),
            kwargs={},
            name=f"size_{size_name}",
            num_runs=args.runs,
        )

    # Generate report
    logger.info("\n" + "="*80)
    logger.info("GENERATING REPORTS")
    logger.info("="*80 + "\n")

    report_path = args.output / "benchmark_report.txt"
    report = benchmark.generate_report(report_path)
    print(report)

    json_path = args.output / "benchmark_results.json"
    benchmark.save_results_json(json_path)

    logger.info(f"\nResults saved to: {args.output}")
    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
