"""
Box detection using multiple methods optimized for Jetson.
Combines YOLO detection with color-based fallback.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class BoxDetection:
    """Detected box/item."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[int, int]
    detection_method: str  # "yolo", "color", "contour"
    class_name: str = "box"

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class BoxDetector:
    """
    Multi-method box detector optimized for inventory tracking.

    Detection methods (in priority order):
    1. Custom-trained YOLO model for cardboard boxes
    2. Pre-trained YOLO for common carried items
    3. Color-based detection for cardboard boxes
    4. Contour-based detection for rectangular objects
    """

    # COCO classes that could be carried items
    CARRIED_ITEM_CLASSES = {
        24: "backpack",
        26: "handbag",
        28: "suitcase",
        39: "bottle",
        73: "book",
    }

    # HSV color ranges for cardboard detection
    CARDBOARD_COLORS = [
        # Brown cardboard
        {"lower": np.array([10, 30, 80]), "upper": np.array([25, 180, 220])},
        # Tan/beige
        {"lower": np.array([15, 20, 100]), "upper": np.array([30, 120, 230])},
        # Light brown
        {"lower": np.array([8, 40, 120]), "upper": np.array([20, 150, 250])},
    ]

    def __init__(
        self,
        custom_model_path: Optional[str] = None,
        general_model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.4,
        min_area: int = 1500,
        max_area: int = 80000,
        use_color_detection: bool = True,
        use_contour_detection: bool = True,
        imgsz: int = 480,
        use_tensorrt: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.use_color_detection = use_color_detection
        self.use_contour_detection = use_contour_detection
        self.imgsz = imgsz

        self.custom_model = None
        self.general_model = None

        # Load models
        self._load_models(custom_model_path, general_model_path, use_tensorrt)

    def _load_models(
        self,
        custom_path: Optional[str],
        general_path: str,
        use_tensorrt: bool
    ):
        """Load YOLO models."""
        try:
            from ultralytics import YOLO

            # Load custom box model if available
            if custom_path and Path(custom_path).exists():
                trt_path = Path(custom_path).with_suffix(".engine")
                if use_tensorrt and trt_path.exists():
                    self.custom_model = YOLO(str(trt_path))
                else:
                    self.custom_model = YOLO(custom_path)
                logger.info(f"Loaded custom box model: {custom_path}")

            # Load general model for carried items
            if Path(general_path).exists():
                self.general_model = YOLO(general_path)
                logger.info(f"Loaded general model: {general_path}")

        except ImportError as e:
            logger.warning(f"YOLO not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")

    def detect(
        self,
        frame: np.ndarray,
        person_bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[BoxDetection]:
        """
        Detect boxes in frame.

        Args:
            frame: BGR image
            person_bboxes: Optional list of person bounding boxes to focus search

        Returns:
            List of BoxDetection objects
        """
        all_detections = []

        # Method 1: Custom YOLO model
        if self.custom_model:
            all_detections.extend(self._detect_yolo_custom(frame))

        # Method 2: General YOLO for carried items
        if self.general_model:
            all_detections.extend(self._detect_yolo_general(frame))

        # Method 3: Color-based cardboard detection
        if self.use_color_detection:
            all_detections.extend(self._detect_by_color(frame, person_bboxes))

        # Method 4: Contour-based detection
        if self.use_contour_detection and len(all_detections) < 3:
            all_detections.extend(self._detect_by_contour(frame, person_bboxes))

        # Remove duplicates (NMS)
        all_detections = self._non_max_suppression(all_detections)

        return all_detections

    def _detect_yolo_custom(self, frame: np.ndarray) -> List[BoxDetection]:
        """Detect using custom-trained box model."""
        detections = []

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            results = self.custom_model(
                frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                verbose=False,
                device=device,
            )

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())

                    x1, y1, x2, y2 = xyxy
                    area = (x2 - x1) * (y2 - y1)

                    if self.min_area <= area <= self.max_area:
                        detections.append(BoxDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            center=((x1 + x2) // 2, (y1 + y2) // 2),
                            detection_method="yolo_custom",
                            class_name="box"
                        ))

        except Exception as e:
            logger.error(f"Custom YOLO detection error: {e}")

        return detections

    def _detect_yolo_general(self, frame: np.ndarray) -> List[BoxDetection]:
        """Detect carried items using general YOLO model."""
        detections = []

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            results = self.general_model(
                frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                classes=list(self.CARRIED_ITEM_CLASSES.keys()),
                verbose=False,
                device=device,
            )

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())

                    x1, y1, x2, y2 = xyxy
                    area = (x2 - x1) * (y2 - y1)

                    if self.min_area <= area <= self.max_area:
                        class_name = self.CARRIED_ITEM_CLASSES.get(cls, "item")
                        detections.append(BoxDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            center=((x1 + x2) // 2, (y1 + y2) // 2),
                            detection_method="yolo_general",
                            class_name=class_name
                        ))

        except Exception as e:
            logger.error(f"General YOLO detection error: {e}")

        return detections

    def _detect_by_color(
        self,
        frame: np.ndarray,
        person_bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[BoxDetection]:
        """Detect cardboard boxes by color in carrying zones."""
        detections = []

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for cardboard colors
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for color_range in self.CARDBOARD_COLORS:
            color_mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
            mask = cv2.bitwise_or(mask, color_mask)

        # Focus on person carrying zones if provided
        if person_bboxes:
            zone_mask = np.zeros_like(mask)
            for x1, y1, x2, y2 in person_bboxes:
                # Carrying zone: middle 60% of person width, bottom 70% height
                zone_x1 = x1 + int((x2 - x1) * 0.2)
                zone_x2 = x1 + int((x2 - x1) * 0.8)
                zone_y1 = y1 + int((y2 - y1) * 0.3)
                zone_y2 = y2
                zone_mask[zone_y1:zone_y2, zone_x1:zone_x2] = 255
            mask = cv2.bitwise_and(mask, zone_mask)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area <= area <= self.max_area):
                continue

            # Check if roughly rectangular
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0

            # Box-like aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if not (0.3 <= aspect_ratio <= 3.0):
                continue

            # Extent should be high for boxes (filled rectangle)
            if extent < 0.5:
                continue

            # Calculate confidence based on extent and aspect ratio
            conf = min(extent, 0.8)

            detections.append(BoxDetection(
                bbox=(x, y, x + w, y + h),
                confidence=conf,
                center=(x + w // 2, y + h // 2),
                detection_method="color",
                class_name="cardboard_box"
            ))

        return detections

    def _detect_by_contour(
        self,
        frame: np.ndarray,
        person_bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[BoxDetection]:
        """Detect rectangular objects by edge contours."""
        detections = []

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Focus on carrying zones
        if person_bboxes:
            zone_mask = np.zeros_like(edges)
            for x1, y1, x2, y2 in person_bboxes:
                zone_x1 = x1 + int((x2 - x1) * 0.1)
                zone_x2 = x1 + int((x2 - x1) * 0.9)
                zone_y1 = y1 + int((y2 - y1) * 0.2)
                zone_y2 = y2
                zone_mask[zone_y1:zone_y2, zone_x1:zone_x2] = 255
            edges = cv2.bitwise_and(edges, zone_mask)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Look for quadrilaterals (4 corners)
            if len(approx) != 4:
                continue

            area = cv2.contourArea(approx)
            if not (self.min_area <= area <= self.max_area):
                continue

            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h if h > 0 else 0

            if not (0.3 <= aspect_ratio <= 3.0):
                continue

            detections.append(BoxDetection(
                bbox=(x, y, x + w, y + h),
                confidence=0.5,
                center=(x + w // 2, y + h // 2),
                detection_method="contour",
                class_name="box"
            ))

        return detections

    def _non_max_suppression(
        self,
        detections: List[BoxDetection],
        iou_threshold: float = 0.5
    ) -> List[BoxDetection]:
        """Remove overlapping detections, keeping higher confidence."""
        if len(detections) <= 1:
            return detections

        # Sort by confidence descending
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        kept = []
        for det in detections:
            # Check IoU with kept detections
            should_keep = True
            for kept_det in kept:
                iou = self._calculate_iou(det.bbox, kept_det.bbox)
                if iou > iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                kept.append(det)

        return kept

    @staticmethod
    def _calculate_iou(
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def detect_carried_boxes(
        self,
        frame: np.ndarray,
        person_bbox: Tuple[int, int, int, int]
    ) -> List[BoxDetection]:
        """
        Detect boxes being carried by a specific person.

        Args:
            frame: BGR image
            person_bbox: Person bounding box

        Returns:
            List of boxes associated with this person
        """
        all_boxes = self.detect(frame, [person_bbox])

        carried = []
        for box in all_boxes:
            if self._is_carried_by(box.bbox, person_bbox):
                carried.append(box)

        return carried

    def _is_carried_by(
        self,
        box_bbox: Tuple[int, int, int, int],
        person_bbox: Tuple[int, int, int, int],
        iou_threshold: float = 0.2
    ) -> bool:
        """Check if box is being carried by person."""
        px1, py1, px2, py2 = person_bbox
        bx1, by1, bx2, by2 = box_bbox

        # Box center
        bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2

        # Check if box center is within person's horizontal extent
        if not (px1 <= bcx <= px2):
            return False

        # Check if box is in carrying zone (lower 70% of person)
        carrying_top = py1 + int((py2 - py1) * 0.3)
        if not (carrying_top <= bcy <= py2):
            return False

        # Also check IoU
        iou = self._calculate_iou(box_bbox, person_bbox)
        if iou < iou_threshold:
            return False

        return True
