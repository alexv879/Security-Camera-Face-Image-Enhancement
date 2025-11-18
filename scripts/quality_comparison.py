#!/usr/bin/env python3
"""Compare image quality before and after enhancement."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from rich.console import Console
from rich.table import Table
from loguru import logger

from face_enhancement.config import AppConfig
from face_enhancement.core.pipeline import EnhancementPipeline
from face_enhancement.utils.quality_metrics import QualityAssessment


def main():
    """Compare image quality."""
    parser = argparse.ArgumentParser(description="Compare image quality before/after enhancement")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("--output", type=Path, help="Save enhanced image (optional)")
    parser.add_argument("--model", default="gfpgan", choices=["gfpgan", "codeformer"])
    parser.add_argument("--upscale", type=int, default=2, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    console = Console()

    console.print("[bold green]Image Quality Comparison[/bold green]\n")

    # Load image
    logger.info(f"Loading image: {args.input}")
    original = cv2.imread(str(args.input))

    if original is None:
        console.print(f"[red]Failed to load image: {args.input}[/red]")
        return

    # Initialize pipeline
    logger.info("Initializing enhancement pipeline...")
    config = AppConfig()
    config.enhancement.model = args.model
    config.enhancement.upscale_factor = args.upscale
    pipeline = EnhancementPipeline(config)

    # Enhance image
    logger.info("Enhancing image...")
    enhanced, metadata = pipeline.process_image(original)

    # Calculate quality metrics
    logger.info("Calculating quality metrics...")
    comparison = QualityAssessment.compare_images(original, enhanced)

    # Display results
    table = Table(title="Quality Metrics Comparison", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Original", style="yellow")
    table.add_column("Enhanced", style="green")
    table.add_column("Improvement", style="magenta")

    for metric in comparison["original"].keys():
        orig_val = comparison["original"][metric]
        enh_val = comparison["enhanced"][metric]
        imp_val = comparison["improvement"].get(metric, 0.0)

        table.add_row(
            metric.upper(),
            f"{orig_val:.4f}",
            f"{enh_val:.4f}",
            f"{imp_val:+.4f}" if imp_val else "-"
        )

    console.print(table)

    # Print summary
    console.print(f"\n[bold]Processing Info:[/bold]")
    console.print(f"  Faces detected: {metadata['num_faces']}")
    console.print(f"  Original shape: {metadata['original_shape']}")
    console.print(f"  Output shape: {metadata['output_shape']}")
    console.print(f"  Model: {args.model}")
    console.print(f"  Upscale: {args.upscale}x")

    # Save if requested
    if args.output:
        cv2.imwrite(str(args.output), enhanced)
        console.print(f"\n[green]✓[/green] Enhanced image saved to: {args.output}")


if __name__ == "__main__":
    main()
