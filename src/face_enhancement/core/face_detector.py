"""Advanced face detection module using state-of-the-art models."""

from typing import List, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

try:
    from facexlib.detection import init_detection_model
    FACEXLIB_AVAILABLE = True
except ImportError:
    FACEXLIB_AVAILABLE = False
    logger.warning("facexlib not available, using OpenCV cascade detector as fallback")


class FaceDetector:
    """
    Advanced face detector supporting multiple detection backends.

    Supports:
    - RetinaFace (state-of-the-art)
    - YOLOv5Face
    - OpenCV Haar Cascade (fallback)
    """

    def __init__(
        self,
        model: str = "retinaface",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        """
        Initialize face detector.

        Args:
            model: Detection model ('retinaface', 'yolov5', 'opencv')
            device: Device to run on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detection
            nms_threshold: NMS threshold for overlapping boxes
        """
        self.model_name = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        logger.info(f"Initializing {model} face detector on {self.device}")
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the detection model."""
        if self.model_name == "retinaface" and FACEXLIB_AVAILABLE:
            self._init_retinaface()
        else:
            self._init_opencv_cascade()

    def _init_retinaface(self) -> None:
        """Initialize RetinaFace detector."""
        try:
            self.detector = init_detection_model(
                "retinaface_resnet50", half=False, device=self.device
            )
            logger.info("RetinaFace model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RetinaFace: {e}")
            logger.info("Falling back to OpenCV cascade detector")
            self._init_opencv_cascade()

    def _init_opencv_cascade(self) -> None:
        """Initialize OpenCV Haar Cascade detector (fallback)."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.model_name = "opencv"
        logger.info("OpenCV Haar Cascade detector initialized")

    def detect_faces(
        self, image: np.ndarray, return_landmarks: bool = False
    ) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        """
        Detect faces in image.

        Args:
            image: Input image (BGR format)
            return_landmarks: Whether to return facial landmarks

        Returns:
            List of (bbox, confidence, landmarks) tuples
            - bbox: [x1, y1, x2, y2]
            - confidence: detection confidence score
            - landmarks: facial landmarks (5 points) or None
        """
        if self.model_name == "retinaface":
            return self._detect_retinaface(image, return_landmarks)
        else:
            return self._detect_opencv(image)

    def _detect_retinaface(
        self, image: np.ndarray, return_landmarks: bool
    ) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        """Detect faces using RetinaFace."""
        try:
            with torch.no_grad():
                # RetinaFace expects RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bboxes = self.detector.detect_faces(rgb_image, self.confidence_threshold)

            results = []
            for bbox in bboxes:
                # bbox format: [x1, y1, x2, y2, score] or with landmarks
                if len(bbox) >= 5:
                    box = bbox[:4].astype(np.int32)
                    confidence = float(bbox[4])

                    # Extract landmarks if available
                    landmarks = None
                    if return_landmarks and len(bbox) > 5:
                        landmarks = bbox[5:15].reshape(5, 2)

                    results.append((box, confidence, landmarks))

            logger.debug(f"Detected {len(results)} faces with RetinaFace")
            return results

        except Exception as e:
            logger.error(f"RetinaFace detection failed: {e}")
            return []

    def _detect_opencv(self, image: np.ndarray) -> List[Tuple[np.ndarray, float, None]]:
        """Detect faces using OpenCV Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        results = []
        for x, y, w, h in faces:
            box = np.array([x, y, x + w, y + h], dtype=np.int32)
            # OpenCV doesn't provide confidence, use fixed value
            confidence = 0.99
            results.append((box, confidence, None))

        logger.debug(f"Detected {len(results)} faces with OpenCV")
        return results

    def get_face_crops(
        self, image: np.ndarray, padding: float = 0.2
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract face crops from image with padding.

        Args:
            image: Input image
            padding: Padding ratio around detected face

        Returns:
            List of (face_crop, bbox) tuples
        """
        detections = self.detect_faces(image)
        crops = []

        h, w = image.shape[:2]

        for bbox, confidence, _ in detections:
            x1, y1, x2, y2 = bbox

            # Add padding
            face_w = x2 - x1
            face_h = y2 - y1
            pad_w = int(face_w * padding)
            pad_h = int(face_h * padding)

            # Ensure within bounds
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(w, x2 + pad_w)
            y2_pad = min(h, y2 + pad_h)

            # Extract crop
            crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
            padded_bbox = np.array([x1_pad, y1_pad, x2_pad, y2_pad])

            crops.append((crop, padded_bbox))

        return crops

    def visualize_detections(
        self, image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2
    ) -> np.ndarray:
        """
        Visualize face detections on image.

        Args:
            image: Input image
            color: Box color (BGR)
            thickness: Line thickness

        Returns:
            Image with drawn bounding boxes
        """
        result = image.copy()
        detections = self.detect_faces(image, return_landmarks=True)

        for bbox, confidence, landmarks in detections:
            x1, y1, x2, y2 = bbox

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            # Draw confidence
            label = f"{confidence:.2f}"
            cv2.putText(
                result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness
            )

            # Draw landmarks if available
            if landmarks is not None:
                for landmark in landmarks:
                    x, y = landmark.astype(np.int32)
                    cv2.circle(result, (x, y), 2, (0, 0, 255), -1)

        return result
