"""Example usage of Face Enhancement AI."""

import cv2
from pathlib import Path

from face_enhancement.core.pipeline import EnhancementPipeline
from face_enhancement.config import AppConfig
from face_enhancement.core.face_detector import FaceDetector
from face_enhancement.preprocessing.image_preprocessor import ImagePreprocessor


def example_basic_enhancement():
    """Basic image enhancement example."""
    print("=== Basic Enhancement Example ===")

    # Initialize pipeline with default config
    config = AppConfig()
    pipeline = EnhancementPipeline(config)

    # Load image
    image = cv2.imread("input.jpg")

    # Process
    enhanced, metadata = pipeline.process_image(image)

    # Save result
    cv2.imwrite("output_enhanced.jpg", enhanced)

    print(f"Detected {metadata['num_faces']} faces")
    print(f"Original shape: {metadata['original_shape']}")
    print(f"Output shape: {metadata['output_shape']}")


def example_custom_config():
    """Custom configuration example."""
    print("\n=== Custom Configuration Example ===")

    # Create custom config
    config = AppConfig()
    config.enhancement.model = "gfpgan"
    config.enhancement.upscale_factor = 4
    config.preprocessing.denoise = True
    config.preprocessing.enhance_brightness = True
    config.preprocessing.gamma_correction = 1.2

    pipeline = EnhancementPipeline(config)

    # Process image
    image = cv2.imread("input.jpg")
    enhanced, metadata = pipeline.process_image(image)

    cv2.imwrite("output_custom.jpg", enhanced)
    print("Custom configuration applied successfully")


def example_face_detection_only():
    """Face detection only example."""
    print("\n=== Face Detection Only Example ===")

    detector = FaceDetector(model="retinaface", device="cuda")

    image = cv2.imread("input.jpg")
    faces = detector.detect_faces(image, return_landmarks=True)

    print(f"Found {len(faces)} faces")

    for i, (bbox, confidence, landmarks) in enumerate(faces):
        print(f"Face {i+1}: bbox={bbox}, confidence={confidence:.2f}")

    # Visualize detections
    visualization = detector.visualize_detections(image)
    cv2.imwrite("output_detections.jpg", visualization)


def example_preprocessing_only():
    """Preprocessing only example."""
    print("\n=== Preprocessing Only Example ===")

    preprocessor = ImagePreprocessor(
        denoise=True,
        auto_contrast=True,
        enhance_brightness=True,
        gamma=1.3,
        sharpen=True
    )

    image = cv2.imread("input.jpg")
    processed = preprocessor.preprocess(image)

    cv2.imwrite("output_preprocessed.jpg", processed)
    print("Preprocessing complete")


def example_batch_processing():
    """Batch processing example."""
    print("\n=== Batch Processing Example ===")

    config = AppConfig()
    pipeline = EnhancementPipeline(config)

    # Get all images in directory
    input_dir = Path("input_images")
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    print(f"Processing {len(image_files)} images...")

    # Process batch
    results = pipeline.process_batch(
        image_files,
        output_dir=Path("output_batch"),
        progress_callback=lambda curr, total: print(f"Progress: {curr}/{total}")
    )

    print(f"Successfully processed {len(results)} images")


def example_video_processing():
    """Video processing example."""
    print("\n=== Video Processing Example ===")

    config = AppConfig()
    pipeline = EnhancementPipeline(config)

    # Process video (extract and enhance frames)
    metadata = pipeline.process_video(
        Path("surveillance.mp4"),
        output_path=Path("output_video"),
        frame_interval=30,  # Process every 30th frame
        max_frames=100      # Process max 100 frames
    )

    print(f"Total frames: {metadata['total_frames']}")
    print(f"Processed frames: {metadata['processed_frames']}")
    print(f"FPS: {metadata['fps']}")


def example_api_usage():
    """Example of using the REST API."""
    print("\n=== API Usage Example ===")
    print("""
    Start the API server:
        face-enhance-server

    Or:
        uvicorn face_enhancement.api.server:app --host 0.0.0.0 --port 8000

    Then use curl or Python requests:

    # Enhance image
    curl -X POST "http://localhost:8000/enhance" \\
        -F "file=@input.jpg" \\
        -F "model=gfpgan" \\
        -F "upscale=2" \\
        --output enhanced.jpg

    # Detect faces
    curl -X POST "http://localhost:8000/detect" \\
        -F "file=@input.jpg"

    # Get pipeline info
    curl http://localhost:8000/info
    """)


def example_cli_usage():
    """Example of CLI usage."""
    print("\n=== CLI Usage Examples ===")
    print("""
    # Single image enhancement
    face-enhance enhance input.jpg --output enhanced.jpg

    # Batch processing
    face-enhance batch ./input_dir --output-dir ./output_dir --pattern "*.jpg"

    # Video processing
    face-enhance video surveillance.mp4 --interval 30 --max-frames 100

    # Custom model and upscale
    face-enhance enhance input.jpg --model gfpgan --upscale 4 --device cuda

    # View pipeline info
    face-enhance info

    # Verbose output
    face-enhance enhance input.jpg -v
    """)


if __name__ == "__main__":
    print("Face Enhancement AI - Example Usage")
    print("=" * 50)

    # Run examples
    # example_basic_enhancement()
    # example_custom_config()
    # example_face_detection_only()
    # example_preprocessing_only()
    # example_batch_processing()
    # example_video_processing()
    example_api_usage()
    example_cli_usage()

    print("\n" + "=" * 50)
    print("Examples complete!")
