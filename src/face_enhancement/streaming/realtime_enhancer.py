"""Real-time streaming face enhancement for video feeds."""

import queue
import threading
import time
from typing import Optional, Callable
import numpy as np
import cv2
from loguru import logger

from face_enhancement.core.pipeline import EnhancementPipeline
from face_enhancement.config import AppConfig


class RealtimeEnhancer:
    """
    Real-time streaming face enhancement.

    Features:
    - Async processing with frame buffering
    - Frame dropping for performance
    - Adaptive quality based on latency
    - Multi-threaded processing
    - Temporal smoothing between frames
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        max_buffer_size: int = 30,
        target_fps: int = 15,
        enable_temporal_smoothing: bool = True,
    ):
        """
        Initialize realtime enhancer.

        Args:
            config: Enhancement configuration
            max_buffer_size: Maximum frames to buffer
            target_fps: Target processing FPS
            enable_temporal_smoothing: Enable smoothing between frames
        """
        self.config = config or AppConfig()
        self.max_buffer_size = max_buffer_size
        self.target_fps = target_fps
        self.enable_temporal_smoothing = enable_temporal_smoothing

        # Processing pipeline
        self.pipeline = EnhancementPipeline(self.config)

        # Frame queues
        self.input_queue = queue.Queue(maxsize=max_buffer_size)
        self.output_queue = queue.Queue(maxsize=max_buffer_size)

        # Processing thread
        self.processing_thread = None
        self.running = False

        # Statistics
        self.stats = {
            "frames_processed": 0,
            "frames_dropped": 0,
            "avg_latency": 0.0,
            "current_fps": 0.0,
        }

        # Previous frame for temporal smoothing
        self.prev_enhanced = None

        logger.info(f"Realtime enhancer initialized (target: {target_fps} FPS)")

    def start(self) -> None:
        """Start processing thread."""
        if self.running:
            logger.warning("Already running")
            return

        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        logger.info("Processing thread started")

    def stop(self) -> None:
        """Stop processing thread."""
        if not self.running:
            return

        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        logger.info("Processing thread stopped")

    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> bool:
        """
        Add frame to processing queue.

        Args:
            frame: Input frame
            frame_id: Frame identifier

        Returns:
            True if frame was queued, False if dropped
        """
        try:
            self.input_queue.put_nowait((frame, frame_id, time.time()))
            return True
        except queue.Full:
            self.stats["frames_dropped"] += 1
            logger.debug(f"Dropped frame {frame_id} (queue full)")
            return False

    def get_enhanced_frame(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, int]]:
        """
        Get enhanced frame from output queue.

        Args:
            timeout: Timeout in seconds

        Returns:
            Tuple of (enhanced_frame, frame_id) or None
        """
        try:
            frame, frame_id = self.output_queue.get(timeout=timeout)
            return frame, frame_id
        except queue.Empty:
            return None

    def _processing_loop(self) -> None:
        """Main processing loop (runs in thread)."""
        logger.info("Processing loop started")

        last_time = time.time()
        frame_times = []

        while self.running:
            try:
                # Get input frame
                frame, frame_id, enqueue_time = self.input_queue.get(timeout=0.1)

                # Process
                start_time = time.time()
                enhanced = self._enhance_frame(frame)

                # Temporal smoothing
                if self.enable_temporal_smoothing and self.prev_enhanced is not None:
                    enhanced = self._apply_temporal_smoothing(enhanced, self.prev_enhanced)

                self.prev_enhanced = enhanced.copy()

                # Calculate latency
                latency = time.time() - enqueue_time
                processing_time = time.time() - start_time

                # Update stats
                self.stats["frames_processed"] += 1
                self.stats["avg_latency"] = (
                    self.stats["avg_latency"] * 0.9 + latency * 0.1
                )

                # Calculate FPS
                frame_times.append(time.time())
                if len(frame_times) > 30:
                    frame_times.pop(0)
                if len(frame_times) > 1:
                    fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                    self.stats["current_fps"] = fps

                # Put result
                try:
                    self.output_queue.put_nowait((enhanced, frame_id))
                except queue.Full:
                    # Drop oldest if queue full
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait((enhanced, frame_id))
                    except:
                        pass

                logger.debug(
                    f"Frame {frame_id}: latency={latency:.3f}s, "
                    f"processing={processing_time:.3f}s, fps={self.stats['current_fps']:.1f}"
                )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")

        logger.info("Processing loop stopped")

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance single frame."""
        enhanced, _ = self.pipeline.process_image(
            frame,
            preprocess=True,
            detect_faces=True,
            enhance=True,
        )
        return enhanced

    def _apply_temporal_smoothing(
        self,
        current: np.ndarray,
        previous: np.ndarray,
        alpha: float = 0.7,
    ) -> np.ndarray:
        """
        Apply temporal smoothing between frames.

        Args:
            current: Current enhanced frame
            previous: Previous enhanced frame
            alpha: Blending factor (higher = more current frame)

        Returns:
            Smoothed frame
        """
        # Ensure same size
        if current.shape != previous.shape:
            previous = cv2.resize(previous, (current.shape[1], current.shape[0]))

        # Blend frames
        smoothed = cv2.addWeighted(current, alpha, previous, 1 - alpha, 0)

        return smoothed

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "frames_processed": 0,
            "frames_dropped": 0,
            "avg_latency": 0.0,
            "current_fps": 0.0,
        }


class VideoStreamEnhancer:
    """
    Video stream enhancement with adaptive quality.

    Automatically adjusts quality based on processing latency
    to maintain target FPS.
    """

    def __init__(
        self,
        video_source: str = "0",
        output_path: Optional[str] = None,
        target_fps: int = 15,
        adaptive_quality: bool = True,
    ):
        """
        Initialize video stream enhancer.

        Args:
            video_source: Video source (camera index or file path)
            output_path: Optional output video path
            target_fps: Target FPS
            adaptive_quality: Enable adaptive quality
        """
        self.video_source = video_source
        self.output_path = output_path
        self.target_fps = target_fps
        self.adaptive_quality = adaptive_quality

        # Open video
        if video_source.isdigit():
            self.cap = cv2.VideoCapture(int(video_source))
        else:
            self.cap = cv2.VideoCapture(video_source)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {video_source}")

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Create enhancer
        config = AppConfig()
        self.enhancer = RealtimeEnhancer(
            config=config,
            target_fps=target_fps,
            enable_temporal_smoothing=True,
        )

        # Video writer
        self.writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                output_path,
                fourcc,
                target_fps,
                (self.frame_width * 2, self.frame_height * 2),  # Upscaled
            )

        logger.info(f"Video stream enhancer initialized: {self.frame_width}x{self.frame_height} @ {self.source_fps} FPS")

    def run(
        self,
        display: bool = True,
        max_frames: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Run enhancement on video stream.

        Args:
            display: Display output in window
            max_frames: Maximum frames to process (None = unlimited)
            callback: Optional callback(frame, enhanced, frame_id)
        """
        self.enhancer.start()

        frame_id = 0
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process frame
                self.enhancer.process_frame(frame, frame_id)

                # Get enhanced frame
                result = self.enhancer.get_enhanced_frame(timeout=0.05)

                if result is not None:
                    enhanced, _ = result

                    # Write to video
                    if self.writer:
                        self.writer.write(enhanced)

                    # Display
                    if display:
                        # Create side-by-side
                        frame_resized = cv2.resize(frame, (enhanced.shape[1], enhanced.shape[0]))
                        combined = np.hstack([frame_resized, enhanced])
                        cv2.imshow('Enhancement', combined)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    # Callback
                    if callback:
                        callback(frame, enhanced, frame_id)

                frame_id += 1

                if max_frames and frame_id >= max_frames:
                    break

                # Adaptive FPS limiting
                if frame_id % 30 == 0:
                    stats = self.enhancer.get_stats()
                    logger.info(
                        f"Processed: {stats['frames_processed']}, "
                        f"Dropped: {stats['frames_dropped']}, "
                        f"FPS: {stats['current_fps']:.1f}, "
                        f"Latency: {stats['avg_latency']:.3f}s"
                    )

        finally:
            self.enhancer.stop()
            self.cap.release()
            if self.writer:
                self.writer.release()
            cv2.destroyAllWindows()

        logger.info("Stream processing complete")
