"""Command-line interface for face enhancement."""

import sys
from pathlib import Path
from typing import List, Optional

import click
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table

from face_enhancement.config import AppConfig, load_config
from face_enhancement.core.pipeline import EnhancementPipeline
from face_enhancement.utils.logger import setup_logger


console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Face Enhancement AI - State-of-the-art face image enhancement for security camera footage.

    Enhance low-quality face images using deep learning models (GFPGAN, CodeFormer).
    """
    pass


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path (default: input_enhanced.ext)",
)
@click.option(
    "--model",
    type=click.Choice(["gfpgan", "codeformer"]),
    default="gfpgan",
    help="Enhancement model",
)
@click.option(
    "--upscale",
    type=int,
    default=2,
    help="Upscaling factor (1-4)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="Processing device",
)
@click.option(
    "--no-preprocess",
    is_flag=True,
    help="Disable preprocessing",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--save-comparison",
    is_flag=True,
    default=True,
    help="Save before/after comparison",
)
@click.option(
    "--save-detections",
    is_flag=True,
    default=True,
    help="Save face detection visualization",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Verbose output",
)
def enhance(
    input_path: str,
    output: Optional[str],
    model: str,
    upscale: int,
    device: str,
    no_preprocess: bool,
    config: Optional[str],
    save_comparison: bool,
    save_detections: bool,
    verbose: bool,
):
    """Enhance a single image."""
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(log_level=log_level)

    console.print("[bold green]Face Enhancement AI[/bold green] - Processing image...\n")

    try:
        # Load configuration
        app_config = load_config(Path(config) if config else None)
        app_config.enhancement.model = model
        app_config.enhancement.upscale_factor = upscale
        app_config.device = device

        # Initialize pipeline
        with console.status("[bold green]Initializing AI models..."):
            pipeline = EnhancementPipeline(app_config)

        # Process image
        input_file = Path(input_path)
        output_file = Path(output) if output else None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Enhancing image...", total=100)

            enhanced, metadata = pipeline.process_image_file(
                input_file,
                output_file,
                save_comparison=save_comparison,
                save_detections=save_detections,
            )

            progress.update(task, completed=100)

        # Display results
        _display_results(metadata)

        console.print(f"\n[bold green]✓[/bold green] Enhancement complete!")

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        if verbose:
            logger.exception(e)
        sys.exit(1)


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    help="Output directory",
)
@click.option(
    "--pattern",
    default="*.jpg",
    help="File pattern to match (e.g., *.jpg, *.png)",
)
@click.option(
    "--model",
    type=click.Choice(["gfpgan", "codeformer"]),
    default="gfpgan",
    help="Enhancement model",
)
@click.option(
    "--upscale",
    type=int,
    default=2,
    help="Upscaling factor (1-4)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="Processing device",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Verbose output",
)
def batch(
    input_dir: str,
    output_dir: Optional[str],
    pattern: str,
    model: str,
    upscale: int,
    device: str,
    config: Optional[str],
    verbose: bool,
):
    """Enhance multiple images in batch."""
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(log_level=log_level)

    console.print("[bold green]Face Enhancement AI[/bold green] - Batch processing...\n")

    try:
        # Load configuration
        app_config = load_config(Path(config) if config else None)
        app_config.enhancement.model = model
        app_config.enhancement.upscale_factor = upscale
        app_config.device = device

        if output_dir:
            app_config.output_dir = Path(output_dir)

        # Find input files
        input_path = Path(input_dir)
        input_files = list(input_path.glob(pattern))

        if not input_files:
            console.print(f"[yellow]No files matching pattern '{pattern}' found in {input_dir}[/yellow]")
            return

        console.print(f"Found {len(input_files)} files to process\n")

        # Initialize pipeline
        with console.status("[bold green]Initializing AI models..."):
            pipeline = EnhancementPipeline(app_config)

        # Process batch
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing images...", total=len(input_files))

            def update_progress(current, total):
                progress.update(task, completed=current)

            results = pipeline.process_batch(
                input_files,
                output_dir=app_config.output_dir,
                progress_callback=update_progress,
            )

        # Display summary
        _display_batch_summary(results)

        console.print(f"\n[bold green]✓[/bold green] Batch processing complete!")

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        if verbose:
            logger.exception(e)
        sys.exit(1)


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    help="Output directory for frames",
)
@click.option(
    "--interval",
    type=int,
    default=30,
    help="Process every Nth frame",
)
@click.option(
    "--max-frames",
    type=int,
    help="Maximum number of frames to process",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Verbose output",
)
def video(
    video_path: str,
    output_dir: Optional[str],
    interval: int,
    max_frames: Optional[int],
    config: Optional[str],
    verbose: bool,
):
    """Extract and enhance frames from video."""
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(log_level=log_level)

    console.print("[bold green]Face Enhancement AI[/bold green] - Processing video...\n")

    try:
        app_config = load_config(Path(config) if config else None)

        with console.status("[bold green]Initializing AI models..."):
            pipeline = EnhancementPipeline(app_config)

        video_file = Path(video_path)
        out_dir = Path(output_dir) if output_dir else None

        metadata = pipeline.process_video(
            video_file,
            output_path=out_dir,
            frame_interval=interval,
            max_frames=max_frames,
        )

        console.print(f"\n[bold green]✓[/bold green] Processed {metadata['processed_frames']} frames")

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        if verbose:
            logger.exception(e)
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def info(config: Optional[str]):
    """Display pipeline configuration and model information."""
    try:
        app_config = load_config(Path(config) if config else None)
        pipeline = EnhancementPipeline(app_config)

        info_data = pipeline.get_pipeline_info()

        table = Table(title="Pipeline Configuration", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Configuration", style="green")

        # Preprocessor
        table.add_row("Preprocessing", "")
        for key, value in info_data["preprocessor"].items():
            table.add_row(f"  {key}", str(value))

        # Detector
        table.add_row("", "")
        table.add_row("Face Detection", "")
        for key, value in info_data["detector"].items():
            table.add_row(f"  {key}", str(value))

        # Enhancer
        table.add_row("", "")
        table.add_row("Face Enhancement", "")
        for key, value in info_data["enhancer"].items():
            table.add_row(f"  {key}", str(value))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        sys.exit(1)


def _display_results(metadata: dict):
    """Display processing results."""
    table = Table(title="Processing Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Faces detected", str(metadata.get("num_faces", 0)))
    table.add_row("Original size", f"{metadata.get('original_shape', [0, 0])[:2]}")
    table.add_row("Output size", f"{metadata.get('output_shape', [0, 0])[:2]}")
    table.add_row("Preprocessing", "Yes" if metadata.get("preprocessing_applied") else "No")
    table.add_row("Enhancement", "Yes" if metadata.get("enhancement_applied") else "No")

    if "output_path" in metadata:
        table.add_row("Output path", str(metadata["output_path"]))

    console.print(table)


def _display_batch_summary(results: List[dict]):
    """Display batch processing summary."""
    total = len(results)
    successful = len([r for r in results if "error" not in r])
    failed = total - successful
    total_faces = sum(r.get("num_faces", 0) for r in results if "error" not in r)

    table = Table(title="Batch Processing Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total images", str(total))
    table.add_row("Successful", str(successful))
    table.add_row("Failed", str(failed))
    table.add_row("Total faces detected", str(total_faces))

    console.print(table)


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
