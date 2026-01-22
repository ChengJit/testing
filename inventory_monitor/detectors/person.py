"""
Person detection using YOLOv11 optimized for Jetson Orin Nano.
Supports TensorRT acceleration for maximum performance.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """Detected person with bounding box and confidence."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[int, int]

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class PersonDetector:
    """
    YOLO-based person detector optimized for Jetson.

    Features:
    - TensorRT acceleration (auto-export on first run)
    - FP16 inference for speed
    - Configurable input resolution
    - Efficient batching
    """

    PERSON_CLASS_ID = 0  # COCO class ID for person

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        imgsz: int = 480,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        use_tensorrt: bool = True,
        fp16: bool = True,
        device: str = "auto"
    ):
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_tensorrt = use_tensorrt
        self.fp16 = fp16

        self._load_model(model_path, device)

    def _load_model(self, model_path: str, device: str):
        """Load YOLO model with optional TensorRT export."""
        try:
            from ultralytics import YOLO

            model_file = Path(model_path)

            # Check for existing TensorRT engine
            trt_path = model_file.with_suffix(".engine")

            if self.use_tensorrt and trt_path.exists():
                logger.info(f"Loading TensorRT engine: {trt_path}")
                self.model = YOLO(str(trt_path))
            else:
                logger.info(f"Loading YOLO model: {model_path}")
                self.model = YOLO(model_path)

                # Export to TensorRT on Jetson
                if self.use_tensorrt:
                    self._export_tensorrt(model_file)

            # Determine device
            if device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            logger.info(f"Person detector initialized on {self.device}")

        except ImportError as e:
            logger.error(f"Failed to import ultralytics: {e}")
            raise

    def _export_tensorrt(self, model_path: Path):
        """Export model to TensorRT format for Jetson acceleration."""
        try:
            import platform
            # Only attempt TensorRT export on Linux (Jetson)
            if platform.system() != "Linux":
                logger.info("Skipping TensorRT export (not on Linux)")
                return

            logger.info("Exporting model to TensorRT (this may take a few minutes)...")
            self.model.export(
                format="engine",
                imgsz=self.imgsz,
                half=self.fp16,
                simplify=True,
                workspace=4,  # 4GB workspace for Jetson
            )

            # Reload the TensorRT model
            trt_path = model_path.with_suffix(".engine")
            if trt_path.exists():
                from ultralytics import YOLO
                self.model = YOLO(str(trt_path))
                logger.info("TensorRT engine loaded successfully")

        except Exception as e:
            logger.warning(f"TensorRT export failed, using PyTorch: {e}")

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Detect persons in frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            List of PersonDetection objects
        """
        detections = []

        try:
            # Run inference
            results = self.model(
                frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[self.PERSON_CLASS_ID],  # Only detect persons
                verbose=False,
                device=self.device,
            )

            # Process results
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        # Get bounding box
                        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())

                        if cls == self.PERSON_CLASS_ID:
                            x1, y1, x2, y2 = xyxy
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)

                            detections.append(PersonDetection(
                                bbox=(x1, y1, x2, y2),
                                confidence=conf,
                                center=center
                            ))

        except Exception as e:
            logger.error(f"Detection error: {e}")

        return detections

    def warmup(self, input_shape: Tuple[int, int, int] = (480, 640, 3)):
        """Warmup the model with a dummy inference."""
        dummy = np.zeros(input_shape, dtype=np.uint8)
        self.detect(dummy)
        logger.info("Person detector warmed up")
