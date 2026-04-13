"""
Zero-Shot Object Detection for Rack Inventory
Supports YOLO-World (fast) and GroundingDINO (accurate)
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class Detection:
    """Single detection result."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    area: int = 0

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.area = (x2 - x1) * (y2 - y1)


@dataclass
class ShelfZone:
    """Defines a shelf zone for counting."""
    name: str
    y1: float  # normalized 0-1
    y2: float
    x1: float = 0.0
    x2: float = 1.0


class YOLOWorldDetector:
    """
    YOLO-World: Fast open-vocabulary object detection.
    Can detect any object by text prompt without training.
    """

    def __init__(
        self,
        model_size: str = "s",  # s, m, l, x
        conf_threshold: float = 0.3,
        device: str = "auto",
    ):
        self.conf_threshold = conf_threshold
        self.model = None
        self.classes = []
        self._model_size = model_size

    def load(self, classes: List[str]):
        """Load model with specified classes to detect."""
        try:
            from ultralytics import YOLO

            model_name = f"yolov8{self._model_size}-worldv2.pt"
            print(f"Loading YOLO-World ({model_name})...")

            self.model = YOLO(model_name)
            self.classes = classes
            self.model.set_classes(classes)

            print(f"  Classes: {classes}")
            return True

        except ImportError:
            print("ERROR: ultralytics not installed")
            print("  pip install ultralytics")
            return False
        except Exception as e:
            print(f"ERROR loading YOLO-World: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in frame."""
        if self.model is None:
            return []

        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                label = self.classes[cls_id] if cls_id < len(self.classes) else "unknown"

                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=tuple(bbox),
                ))

        return detections


class GroundingDINODetector:
    """
    GroundingDINO: Accurate text-guided object detection.
    Slower but more accurate than YOLO-World.
    """

    def __init__(
        self,
        conf_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.conf_threshold = conf_threshold
        self.text_threshold = text_threshold
        self.model = None
        self.prompt = ""

    def load(self, classes: List[str]):
        """Load model with text prompt."""
        try:
            # Try groundingdino package
            from groundingdino.util.inference import load_model, predict

            # Model paths (adjust as needed)
            config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            weights_path = "weights/groundingdino_swint_ogc.pth"

            print("Loading GroundingDINO...")
            self.model = load_model(config_path, weights_path)
            self.prompt = " . ".join(classes) + " ."
            self._predict_fn = predict

            print(f"  Prompt: {self.prompt}")
            return True

        except ImportError:
            print("GroundingDINO not installed. Using YOLO-World instead.")
            return False
        except Exception as e:
            print(f"ERROR loading GroundingDINO: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects using text prompt."""
        if self.model is None:
            return []

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        boxes, logits, phrases = self._predict_fn(
            model=self.model,
            image=image_rgb,
            caption=self.prompt,
            box_threshold=self.conf_threshold,
            text_threshold=self.text_threshold,
        )

        h, w = frame.shape[:2]
        detections = []

        for box, conf, phrase in zip(boxes, logits, phrases):
            # Convert normalized coords to pixel coords
            cx, cy, bw, bh = box.cpu().numpy()
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            detections.append(Detection(
                label=phrase,
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
            ))

        return detections


class RackInventoryDetector:
    """
    Main detector for rack inventory.
    Combines detection with shelf zone counting.
    """

    def __init__(
        self,
        use_yolo_world: bool = True,
        model_size: str = "s",
        conf_threshold: float = 0.3,
    ):
        self.use_yolo_world = use_yolo_world

        if use_yolo_world:
            self.detector = YOLOWorldDetector(
                model_size=model_size,
                conf_threshold=conf_threshold,
            )
        else:
            self.detector = GroundingDINODetector(
                conf_threshold=conf_threshold,
            )

        self.shelf_zones: List[ShelfZone] = []
        self.loaded = False

    def load(self, classes: List[str] = None):
        """Load detector with classes to detect."""
        if classes is None:
            classes = [
                "cardboard box",
                "box",
                "terminal",
                "device",
                "package",
            ]

        self.loaded = self.detector.load(classes)
        return self.loaded

    def set_shelf_zones(self, zones: List[ShelfZone]):
        """Define shelf zones for counting."""
        self.shelf_zones = zones

    def auto_detect_shelves(self, frame: np.ndarray) -> List[ShelfZone]:
        """
        Auto-detect shelf levels based on yellow rack bars.
        Returns list of shelf zones.
        """
        # Convert to HSV for yellow detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Yellow color range (rack bars)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Find horizontal lines (shelf bars)
        h, w = frame.shape[:2]
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 3))
        detected_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)

        # Find contours of horizontal bars
        contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get Y coordinates of shelf bars
        shelf_ys = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > w * 0.3:  # Only wide bars
                shelf_ys.append(y + ch // 2)

        shelf_ys = sorted(set(shelf_ys))

        # Create zones between shelf bars
        zones = []
        for i in range(len(shelf_ys) - 1):
            y1_norm = shelf_ys[i] / h
            y2_norm = shelf_ys[i + 1] / h

            # Skip very thin zones
            if y2_norm - y1_norm > 0.05:
                zones.append(ShelfZone(
                    name=f"Shelf-{i+1}",
                    y1=y1_norm,
                    y2=y2_norm,
                ))

        return zones

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], Dict[str, int]]:
        """
        Detect objects and count by zone.

        Returns:
            detections: List of all detections
            counts: Dict of zone_name -> count
        """
        if not self.loaded:
            return [], {}

        detections = self.detector.detect(frame)
        h, w = frame.shape[:2]

        # Count by zone
        counts = {}
        for zone in self.shelf_zones:
            zone_y1 = int(zone.y1 * h)
            zone_y2 = int(zone.y2 * h)
            zone_x1 = int(zone.x1 * w)
            zone_x2 = int(zone.x2 * w)

            count = 0
            for det in detections:
                # Check if detection center is in zone
                cx = (det.bbox[0] + det.bbox[2]) // 2
                cy = (det.bbox[1] + det.bbox[3]) // 2

                if zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2:
                    count += 1

            counts[zone.name] = count

        # Total count
        counts["Total"] = len(detections)

        return detections, counts

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        counts: Dict[str, int] = None,
    ) -> np.ndarray:
        """Draw detections and counts on frame."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw shelf zones
        for zone in self.shelf_zones:
            y1 = int(zone.y1 * h)
            y2 = int(zone.y2 * h)
            cv2.rectangle(display, (0, y1), (w, y2), (255, 255, 0), 1)
            cv2.putText(display, zone.name, (5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)  # Green

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"{det.label} {det.confidence:.0%}"
            cv2.putText(display, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw counts panel
        if counts:
            panel_h = 25 + len(counts) * 22
            cv2.rectangle(display, (w - 150, 0), (w, panel_h), (0, 0, 0), -1)
            cv2.putText(display, "COUNTS", (w - 140, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y = 42
            for zone, count in counts.items():
                cv2.putText(display, f"{zone}: {count}", (w - 140, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y += 22

        return display


def test_detector():
    """Test detector on a single image."""
    import sys

    detector = RackInventoryDetector(use_yolo_world=True, model_size="s")
    detector.load([
        "cardboard box",
        "box",
        "package",
        "terminal",
        "device",
    ])

    # Test on webcam or image
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
    else:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

    if frame is None:
        print("No image to test")
        return

    # Auto-detect shelves
    zones = detector.auto_detect_shelves(frame)
    print(f"Detected {len(zones)} shelf zones")
    detector.set_shelf_zones(zones)

    # Detect
    start = time.time()
    detections, counts = detector.detect(frame)
    elapsed = (time.time() - start) * 1000

    print(f"Detection time: {elapsed:.0f}ms")
    print(f"Found {len(detections)} objects")
    print(f"Counts: {counts}")

    # Display
    display = detector.draw_detections(frame, detections, counts)
    cv2.imshow("Detection Test", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_detector()
