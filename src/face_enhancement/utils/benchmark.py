"""Performance benchmarking utilities."""

import time
import numpy as np
import cv2
from typing import Dict, List, Callable
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_time: float
    throughput: float  # images per second
    memory_used: float  # MB
    gpu_memory_used: float  # MB (if available)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking for face enhancement pipeline.

    Measures:
    - Processing time
    - Throughput
    - Memory usage
    - GPU utilization
    - Quality metrics
    """

    def __init__(self):
        """Initialize benchmark."""
        self.results: List[BenchmarkResult] = []

    def benchmark_function(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        name: str,
        num_runs: int = 10,
        warmup_runs: int = 2,
    ) -> BenchmarkResult:
        """
        Benchmark a function.

        Args:
            func: Function to benchmark
            args: Positional arguments
            kwargs: Keyword arguments
            name: Benchmark name
            num_runs: Number of runs
            warmup_runs: Number of warmup runs

        Returns:
            BenchmarkResult
        """
        logger.info(f"Benchmarking: {name}")

        # Warmup
        for _ in range(warmup_runs):
            func(*args, **kwargs)

        # Benchmark
        times = []
        for i in range(num_runs):
            start_time = time.time()
            func(*args, **kwargs)
            elapsed = time.time() - start_time
            times.append(elapsed)
            logger.debug(f"Run {i+1}/{num_runs}: {elapsed:.4f}s")

        # Calculate statistics
        times_arr = np.array(times)
        total_time = np.sum(times_arr)
        avg_time = np.mean(times_arr)
        min_time = np.min(times_arr)
        max_time = np.max(times_arr)
        std_time = np.std(times_arr)
        throughput = num_runs / total_time

        # Memory usage (simplified)
        import psutil
        process = psutil.Process()
        memory_used = process.memory_info().rss / 1024 / 1024  # MB

        # GPU memory (if available)
        gpu_memory = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        except ImportError:
            pass

        result = BenchmarkResult(
            name=name,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_time=std_time,
            throughput=throughput,
            memory_used=memory_used,
            gpu_memory_used=gpu_memory,
        )

        self.results.append(result)

        logger.info(f"Results: avg={avg_time:.4f}s, throughput={throughput:.2f} img/s")

        return result

    def benchmark_pipeline(
        self,
        pipeline,
        test_images: List[np.ndarray],
        variations: Dict[str, dict] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark entire pipeline with different configurations.

        Args:
            pipeline: EnhancementPipeline instance
            test_images: List of test images
            variations: Dict of configuration variations

        Returns:
            Dict of benchmark results
        """
        results = {}

        if variations is None:
            variations = {"default": {}}

        for var_name, config in variations.items():
            logger.info(f"Benchmarking variation: {var_name}")

            # Apply configuration
            for key, value in config.items():
                setattr(pipeline.config, key, value)

            # Benchmark
            def process_batch():
                for img in test_images:
                    pipeline.process_image(img)

            result = self.benchmark_function(
                process_batch,
                args=(),
                kwargs={},
                name=f"pipeline_{var_name}",
                num_runs=5,
            )

            results[var_name] = result

        return results

    def generate_report(self, output_path: Path = None) -> str:
        """
        Generate benchmark report.

        Args:
            output_path: Optional path to save report

        Returns:
            Report string
        """
        report_lines = [
            "=" * 80,
            "PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            "",
        ]

        for result in self.results:
            report_lines.extend([
                f"Benchmark: {result.name}",
                "-" * 80,
                f"  Total Time:    {result.total_time:.4f}s",
                f"  Average Time:  {result.avg_time:.4f}s",
                f"  Min Time:      {result.min_time:.4f}s",
                f"  Max Time:      {result.max_time:.4f}s",
                f"  Std Dev:       {result.std_time:.4f}s",
                f"  Throughput:    {result.throughput:.2f} images/s",
                f"  Memory Used:   {result.memory_used:.2f} MB",
                f"  GPU Memory:    {result.gpu_memory_used:.2f} MB",
                "",
            ])

        report = "\n".join(report_lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")

        return report

    def save_results_json(self, output_path: Path) -> None:
        """Save results as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_benchmarks": len(self.results),
                "avg_throughput": np.mean([r.throughput for r in self.results]),
            }
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    def compare_models(
        self,
        models: Dict[str, Callable],
        test_image: np.ndarray,
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare different models on the same image.

        Args:
            models: Dict of {name: model_function}
            test_image: Test image

        Returns:
            Dict of benchmark results
        """
        results = {}

        for name, model_func in models.items():
            result = self.benchmark_function(
                model_func,
                args=(test_image,),
                kwargs={},
                name=name,
                num_runs=10,
            )
            results[name] = result

        return results


def create_test_images(sizes: List[tuple] = None) -> List[np.ndarray]:
    """
    Create synthetic test images for benchmarking.

    Args:
        sizes: List of (height, width) tuples

    Returns:
        List of test images
    """
    if sizes is None:
        sizes = [(480, 640), (720, 1280), (1080, 1920), (2160, 3840)]

    images = []
    for h, w in sizes:
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        images.append(img)

    return images
