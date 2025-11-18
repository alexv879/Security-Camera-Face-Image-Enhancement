#!/usr/bin/env python3
"""Real-time video stream enhancement script."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from face_enhancement.streaming.realtime_enhancer import VideoStreamEnhancer
from loguru import logger


def main():
    """Run real-time stream enhancement."""
    parser = argparse.ArgumentParser(description="Real-time video stream enhancement")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source (camera index or file path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output video path (optional)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Target FPS",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frames to process",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window",
    )
    args = parser.parse_args()

    logger.info("Starting real-time video enhancement")
    logger.info(f"Source: {args.source}")
    logger.info(f"Target FPS: {args.fps}")

    if args.output:
        logger.info(f"Output: {args.output}")

    try:
        # Create enhancer
        enhancer = VideoStreamEnhancer(
            video_source=args.source,
            output_path=str(args.output) if args.output else None,
            target_fps=args.fps,
            adaptive_quality=True,
        )

        # Run
        enhancer.run(
            display=not args.no_display,
            max_frames=args.max_frames,
        )

        logger.info("Enhancement complete!")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
