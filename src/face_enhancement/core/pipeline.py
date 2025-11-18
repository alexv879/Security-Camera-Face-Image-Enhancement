"""Complete face enhancement pipeline."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from face_enhancement.config import AppConfig
from face_enhancement.core.face_detector import FaceDetector
from face_enhancement.models.face_enhancer import FaceEnhancer
from face_enhancement.preprocessing.image_preprocessor import ImagePreprocessor
from face_enhancement.utils.image_utils import create_comparison_image, save_image


class EnhancementPipeline:
    """
    Complete end-to-end face enhancement pipeline.

    Pipeline stages:
    1. Preprocessing (denoising, brightness, contrast)
    2. Face detection
    3. Face enhancement (GFPGAN/CodeFormer)
    4. Post-processing
    5. Output generation
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Application configuration
        """
        self.config = config or AppConfig()

        # Initialize components
        logger.info("Initializing enhancement pipeline")

        self.preprocessor = ImagePreprocessor(
            denoise=self.config.preprocessing.denoise,
            auto_contrast=self.config.preprocessing.auto_contrast,
            enhance_brightness=self.config.preprocessing.enhance_brightness,
            gamma=self.config.preprocessing.gamma_correction,
            sharpen=self.config.preprocessing.sharpen,
        )

        self.detector = FaceDetector(
            model=self.config.face_detection.model,
            device=self.config.get_device(),
            confidence_threshold=self.config.face_detection.confidence_threshold,
            nms_threshold=self.config.face_detection.nms_threshold,
        )

        self.enhancer = FaceEnhancer(
            model=self.config.enhancement.model,
            upscale=self.config.enhancement.upscale_factor,
            device=self.config.get_device(),
            bg_upsampler=self.config.enhancement.bg_upsampler,
            fidelity_weight=self.config.enhancement.weight,
        )

        logger.info("Pipeline initialized successfully")

    def process_image(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        detect_faces: bool = True,
        enhance: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single image through the pipeline.

        Args:
            image: Input image (BGR)
            preprocess: Apply preprocessing
            detect_faces: Detect faces
            enhance: Apply enhancement

        Returns:
            Tuple of (enhanced_image, metadata)
        """
        metadata = {
            "original_shape": image.shape,
            "num_faces": 0,
            "preprocessing_applied": preprocess,
            "enhancement_applied": enhance,
        }

        result = image.copy()

        # Stage 1: Preprocessing
        if preprocess:
            logger.debug("Applying preprocessing")
            result = self.preprocessor.preprocess(result)
            metadata["preprocessing_applied"] = True

        # Stage 2: Face detection
        faces = []
        if detect_faces:
            logger.debug("Detecting faces")
            faces = self.detector.detect_faces(result)
            metadata["num_faces"] = len(faces)
            logger.info(f"Detected {len(faces)} faces")

        # Stage 3: Enhancement
        if enhance:
            logger.debug("Applying enhancement")
            result, _ = self.enhancer.enhance(
                result,
                only_center_face=False,
            )
            metadata["enhancement_applied"] = True

        metadata["output_shape"] = result.shape

        return result, metadata

    def process_image_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        save_comparison: bool = True,
        save_detections: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process image from file.

        Args:
            input_path: Path to input image
            output_path: Optional output path
            save_comparison: Save before/after comparison
            save_detections: Save detection visualization

        Returns:
            Tuple of (enhanced_image, metadata)
        """
        logger.info(f"Processing image: {input_path}")

        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Failed to load image: {input_path}")

        # Process
        enhanced, metadata = self.process_image(image)

        # Generate output path if not provided
        if output_path is None:
            output_path = (
                self.config.output_dir
                / f"{input_path.stem}_enhanced{input_path.suffix}"
            )

        # Save enhanced image
        save_image(enhanced, output_path, quality=self.config.processing.output_quality)
        logger.info(f"Saved enhanced image to: {output_path}")

        # Save comparison
        if save_comparison:
            comparison = create_comparison_image(image, enhanced)
            comparison_path = output_path.parent / f"{output_path.stem}_comparison.png"
            save_image(comparison, comparison_path)
            logger.info(f"Saved comparison to: {comparison_path}")

        # Save detections
        if save_detections:
            detection_vis = self.detector.visualize_detections(image)
            detection_path = output_path.parent / f"{output_path.stem}_detections.png"
            save_image(detection_vis, detection_path)
            logger.info(f"Saved detections to: {detection_path}")

        metadata["input_path"] = str(input_path)
        metadata["output_path"] = str(output_path)

        return enhanced, metadata

    def process_batch(
        self,
        input_paths: List[Path],
        output_dir: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
    ) -> List[Dict]:
        """
        Process multiple images in batch.

        Args:
            input_paths: List of input image paths
            output_dir: Output directory
            progress_callback: Optional callback for progress updates

        Returns:
            List of metadata dictionaries
        """
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, input_path in enumerate(input_paths):
            logger.info(f"Processing {i+1}/{len(input_paths)}: {input_path.name}")

            try:
                output_path = output_dir / f"{input_path.stem}_enhanced{input_path.suffix}"
                _, metadata = self.process_image_file(
                    input_path,
                    output_path,
                    save_comparison=self.config.processing.save_comparison,
                    save_detections=self.config.processing.save_detections,
                )
                results.append(metadata)

                if progress_callback:
                    progress_callback(i + 1, len(input_paths))

            except Exception as e:
                logger.error(f"Failed to process {input_path}: {e}")
                results.append({"input_path": str(input_path), "error": str(e)})

        return results

    def process_video(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        frame_interval: int = 1,
        max_frames: Optional[int] = None,
    ) -> Dict:
        """
        Process video file (extract and enhance frames).

        Args:
            video_path: Path to input video
            output_path: Optional output directory for frames
            frame_interval: Process every Nth frame
            max_frames: Maximum number of frames to process

        Returns:
            Metadata dictionary
        """
        logger.info(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        output_dir = output_path or (self.config.output_dir / video_path.stem)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "processed_frames": 0,
        }

        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                logger.info(f"Processing frame {frame_count}")

                enhanced, _ = self.process_image(frame)

                # Save frame
                frame_path = output_dir / f"frame_{frame_count:06d}_enhanced.png"
                save_image(enhanced, frame_path)

                processed_count += 1

                if max_frames and processed_count >= max_frames:
                    break

            frame_count += 1

        cap.release()

        metadata["processed_frames"] = processed_count
        logger.info(f"Processed {processed_count} frames from video")

        return metadata

    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline configuration."""
        return {
            "preprocessor": {
                "denoise": self.preprocessor.denoise,
                "auto_contrast": self.preprocessor.auto_contrast,
                "enhance_brightness": self.preprocessor.enhance_brightness,
                "gamma": self.preprocessor.gamma,
                "sharpen": self.preprocessor.sharpen,
            },
            "detector": {
                "model": self.detector.model_name,
                "device": self.detector.device,
                "confidence_threshold": self.detector.confidence_threshold,
            },
            "enhancer": self.enhancer.get_model_info(),
            "device": self.config.get_device(),
        }
