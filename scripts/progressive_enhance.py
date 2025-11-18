#!/usr/bin/env python3
"""Progressive multi-stage enhancement script."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from rich.console import Console
from loguru import logger

from face_enhancement.models.progressive_enhancer import ProgressiveEnhancer
from face_enhancement.utils.visualization import EnhancementVisualizer


def main():
    """Run progressive enhancement."""
    parser = argparse.ArgumentParser(description="Progressive multi-stage enhancement")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("--output", type=Path, help="Output image path")
    parser.add_argument(
        "--stages",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="Number of enhancement stages",
    )
    parser.add_argument(
        "--model",
        default="gfpgan",
        choices=["gfpgan", "codeformer"],
        help="Base enhancement model",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Processing device",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization of all stages",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional reference image for guided enhancement",
    )
    args = parser.parse_args()

    console = Console()
    console.print("[bold green]Progressive Multi-Stage Enhancement[/bold green]\n")

    # Load input
    logger.info(f"Loading input: {args.input}")
    image = cv2.imread(str(args.input))

    if image is None:
        console.print(f"[red]Failed to load image: {args.input}[/red]")
        return

    # Create enhancer
    logger.info(f"Creating {args.stages}-stage progressive enhancer")
    enhancer = ProgressiveEnhancer(
        base_model=args.model,
        stages=args.stages,
        device=args.device,
    )

    # Print stage info
    stage_info = enhancer.get_stage_info()
    console.print(f"Stages: {stage_info['num_stages']}")
    console.print(f"Upscale factors: {stage_info['stage_upscales']}")
    console.print(f"Fidelity weights: {[f'{w:.2f}' for w in stage_info['stage_fidelity']]}\n")

    # Load reference if provided
    reference = None
    if args.reference:
        logger.info(f"Loading reference: {args.reference}")
        reference = cv2.imread(str(args.reference))

    # Enhance
    console.print("Enhancing...", style="bold yellow")

    if reference is not None:
        enhanced, metadata = enhancer.enhance_with_guidance(
            image,
            reference=reference,
        )
        console.print("[green]✓[/green] Enhanced with reference guidance")
    else:
        enhanced, metadata = enhancer.enhance(
            image,
            intermediate_outputs=args.visualize,
        )
        console.print("[green]✓[/green] Enhancement complete")

    # Save output
    if args.output:
        output_path = args.output
    else:
        output_path = args.input.parent / f"{args.input.stem}_progressive{args.input.suffix}"

    cv2.imwrite(str(output_path), enhanced)
    console.print(f"[green]✓[/green] Saved to: {output_path}")

    # Visualize stages
    if args.visualize and metadata.get("stage_outputs"):
        vis_path = output_path.parent / f"{output_path.stem}_stages.png"

        logger.info("Creating stage visualization")
        visualizer = EnhancementVisualizer()
        vis = visualizer.visualize_progressive_stages(
            metadata["stage_outputs"],
            output_path=vis_path,
        )

        console.print(f"[green]✓[/green] Stage visualization saved to: {vis_path}")

    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
