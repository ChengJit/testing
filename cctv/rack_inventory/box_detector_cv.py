#!/usr/bin/env python3
"""
Box Detection Methods Comparison
=================================

Multiple detection methods for terminal boxes:
1. YOLO-World (AI, zero-shot)
2. GroundingDINO (AI, more accurate)
3. Edge Detection (Traditional CV, fast)
4. Color Segmentation (Traditional CV, for cardboard)
5. Hybrid (Color + Edge)

For uniform boxes, traditional CV often works better!
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class DetectedBox:
    """Detected box with bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    method: str

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class EdgeBasedDetector:
    """
    Detect boxes using edge detection.
    Good for uniform boxes with clear edges.
    """

    def __init__(
        self,
        min_area: int = 800,
        max_area: int = 20000,
        min_aspect: float = 0.3,
        max_aspect: float = 3.0,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    def detect(self, frame: np.ndarray, roi: Tuple[float, float, float, float] = None) -> List[DetectedBox]:
        """Detect boxes using Canny edge detection."""
        h, w = frame.shape[:2]

        # Apply ROI if specified
        if roi:
            x1 = int(roi[0] * w)
            y1 = int(roi[1] * h)
            x2 = int(roi[2] * w)
            y2 = int(roi[3] * h)
            work_frame = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            work_frame = frame
            offset = (0, 0)

        # Grayscale
        gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            # Get bounding rectangle
            x, y, bw, bh = cv2.boundingRect(cnt)

            # Check aspect ratio
            aspect = bw / bh if bh > 0 else 0
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            # Add offset for ROI
            boxes.append(DetectedBox(
                x1=x + offset[0],
                y1=y + offset[1],
                x2=x + bw + offset[0],
                y2=y + bh + offset[1],
                confidence=0.8,
                method="edge"
            ))

        return boxes


class ColorBasedDetector:
    """
    Detect boxes using color segmentation.
    Good for brown cardboard boxes.
    """

    def __init__(
        self,
        # HSV range for brown cardboard
        lower_hsv: Tuple[int, int, int] = (5, 30, 80),
        upper_hsv: Tuple[int, int, int] = (25, 180, 220),
        min_area: int = 800,
        max_area: int = 20000,
    ):
        self.lower_hsv = np.array(lower_hsv)
        self.upper_hsv = np.array(upper_hsv)
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame: np.ndarray, roi: Tuple[float, float, float, float] = None) -> List[DetectedBox]:
        """Detect brown cardboard boxes."""
        h, w = frame.shape[:2]

        # Apply ROI
        if roi:
            x1 = int(roi[0] * w)
            y1 = int(roi[1] * h)
            x2 = int(roi[2] * w)
            y2 = int(roi[3] * h)
            work_frame = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            work_frame = frame
            offset = (0, 0)

        # Convert to HSV
        hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)

        # Create mask for brown color
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            boxes.append(DetectedBox(
                x1=x + offset[0],
                y1=y + offset[1],
                x2=x + bw + offset[0],
                y2=y + bh + offset[1],
                confidence=0.7,
                method="color"
            ))

        return boxes


class HybridDetector:
    """
    Combines edge + color detection for better accuracy.
    """

    def __init__(self):
        self.edge_detector = EdgeBasedDetector()
        self.color_detector = ColorBasedDetector()

    def detect(self, frame: np.ndarray, roi: Tuple[float, float, float, float] = None) -> List[DetectedBox]:
        """Detect using both methods and merge results."""
        edge_boxes = self.edge_detector.detect(frame, roi)
        color_boxes = self.color_detector.detect(frame, roi)

        # Merge overlapping boxes
        all_boxes = edge_boxes + color_boxes
        merged = self._merge_overlapping(all_boxes)

        return merged

    def _merge_overlapping(self, boxes: List[DetectedBox], iou_thresh: float = 0.3) -> List[DetectedBox]:
        """Merge overlapping detections."""
        if not boxes:
            return []

        # Sort by area (larger first)
        boxes = sorted(boxes, key=lambda b: b.area, reverse=True)

        merged = []
        used = set()

        for i, box1 in enumerate(boxes):
            if i in used:
                continue

            # Find overlapping boxes
            group = [box1]
            for j, box2 in enumerate(boxes[i+1:], i+1):
                if j in used:
                    continue
                if self._iou(box1, box2) > iou_thresh:
                    group.append(box2)
                    used.add(j)

            # Average the group
            avg_x1 = int(np.mean([b.x1 for b in group]))
            avg_y1 = int(np.mean([b.y1 for b in group]))
            avg_x2 = int(np.mean([b.x2 for b in group]))
            avg_y2 = int(np.mean([b.y2 for b in group]))

            merged.append(DetectedBox(
                x1=avg_x1, y1=avg_y1, x2=avg_x2, y2=avg_y2,
                confidence=0.9 if len(group) > 1 else 0.7,
                method="hybrid"
            ))
            used.add(i)

        return merged

    def _iou(self, box1: DetectedBox, box2: DetectedBox) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection

        return intersection / union if union > 0 else 0.0


class GroundingDINODetector:
    """
    GroundingDINO for accurate text-guided detection.
    More accurate than YOLO-World but slower.
    """

    def __init__(self, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.model = None
        self.processor = None

    def load(self):
        """Load GroundingDINO model."""
        try:
            # Try transformers implementation (easier to install)
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            import torch

            print("Loading GroundingDINO (this may take a moment)...")
            model_id = "IDEA-Research/grounding-dino-tiny"

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("  Using GPU")
            else:
                print("  Using CPU (slower)")

            print("  GroundingDINO ready")
            return True

        except ImportError:
            print("GroundingDINO requires: pip install transformers torch")
            return False
        except Exception as e:
            print(f"GroundingDINO error: {e}")
            return False

    def detect(self, frame: np.ndarray, prompt: str = "cardboard box", roi: Tuple[float, float, float, float] = None) -> List[DetectedBox]:
        """Detect objects matching text prompt."""
        if self.model is None:
            return []

        import torch
        from PIL import Image

        h, w = frame.shape[:2]

        # Apply ROI
        if roi:
            x1 = int(roi[0] * w)
            y1 = int(roi[1] * h)
            x2 = int(roi[2] * w)
            y2 = int(roi[3] * h)
            work_frame = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            work_frame = frame
            offset = (0, 0)

        # Convert to PIL Image
        image = Image.fromarray(cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB))

        # Process
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.conf_threshold,
            text_threshold=self.conf_threshold,
            target_sizes=[(image.height, image.width)]
        )[0]

        boxes = []
        for box, score in zip(results["boxes"], results["scores"]):
            bx1, by1, bx2, by2 = box.cpu().numpy().astype(int)

            boxes.append(DetectedBox(
                x1=bx1 + offset[0],
                y1=by1 + offset[1],
                x2=bx2 + offset[0],
                y2=by2 + offset[1],
                confidence=float(score),
                method="grounding_dino"
            ))

        return boxes


class MultiMethodDetector:
    """
    Compare multiple detection methods side by side.
    """

    def __init__(self):
        self.methods = {
            "edge": EdgeBasedDetector(),
            "color": ColorBasedDetector(),
            "hybrid": HybridDetector(),
        }

        # Optional AI methods
        self.yolo_world = None
        self.grounding_dino = None

    def load_yolo_world(self):
        """Load YOLO-World detector."""
        try:
            from ultralytics import YOLO
            print("Loading YOLO-World...")
            self.yolo_world = YOLO("yolov8s-worldv2.pt")
            self.yolo_world.set_classes(["cardboard box", "box"])
            print("  YOLO-World ready")
            return True
        except Exception as e:
            print(f"YOLO-World error: {e}")
            return False

    def load_grounding_dino(self):
        """Load GroundingDINO detector."""
        self.grounding_dino = GroundingDINODetector()
        return self.grounding_dino.load()

    def detect_all(self, frame: np.ndarray, roi: Tuple[float, float, float, float] = None) -> dict:
        """Run all detection methods and return results."""
        results = {}

        # Traditional CV methods
        for name, detector in self.methods.items():
            start = time.time()
            boxes = detector.detect(frame, roi)
            elapsed = (time.time() - start) * 1000
            results[name] = {"boxes": boxes, "time_ms": elapsed}

        # YOLO-World
        if self.yolo_world:
            start = time.time()
            h, w = frame.shape[:2]
            if roi:
                x1, y1 = int(roi[0] * w), int(roi[1] * h)
                x2, y2 = int(roi[2] * w), int(roi[3] * h)
                work_frame = frame[y1:y2, x1:x2]
                offset = (x1, y1)
            else:
                work_frame = frame
                offset = (0, 0)

            yolo_results = self.yolo_world.predict(work_frame, conf=0.25, verbose=False)
            boxes = []
            for r in yolo_results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    bbox = r.boxes.xyxy[i].cpu().numpy().astype(int)
                    boxes.append(DetectedBox(
                        x1=int(bbox[0]) + offset[0],
                        y1=int(bbox[1]) + offset[1],
                        x2=int(bbox[2]) + offset[0],
                        y2=int(bbox[3]) + offset[1],
                        confidence=float(r.boxes.conf[i]),
                        method="yolo_world"
                    ))
            elapsed = (time.time() - start) * 1000
            results["yolo_world"] = {"boxes": boxes, "time_ms": elapsed}

        # GroundingDINO
        if self.grounding_dino and self.grounding_dino.model:
            start = time.time()
            boxes = self.grounding_dino.detect(frame, "cardboard box", roi)
            elapsed = (time.time() - start) * 1000
            results["grounding_dino"] = {"boxes": boxes, "time_ms": elapsed}

        return results


def compare_methods(camera_ip: str):
    """Interactive comparison of detection methods."""
    import sys
    import os
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

    # Build URL
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2?rtsp_transport=tcp"

    print(f"Connecting to {camera_ip}...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Failed to connect!")
        return

    # Create multi-method detector
    detector = MultiMethodDetector()
    detector.load_yolo_world()

    print("\nMethods available:")
    print("  1 - Edge Detection (fast)")
    print("  2 - Color Segmentation (fast)")
    print("  3 - Hybrid (edge + color)")
    print("  4 - YOLO-World (AI)")
    print("  5 - GroundingDINO (AI, accurate - press to load)")
    print("  A - All methods comparison")
    print("\nControls:")
    print("  R - Draw ROI")
    print("  S - Screenshot")
    print("  Q - Quit")
    print("-" * 50)

    roi = None
    current_method = "hybrid"
    show_all = False

    window = "Box Detection Comparison"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            continue

        if show_all:
            # Compare all methods
            results = detector.detect_all(frame, roi)
            display = create_comparison_view(frame, results, roi)
        else:
            # Single method
            if current_method in detector.methods:
                boxes = detector.methods[current_method].detect(frame, roi)
            elif current_method == "yolo_world" and detector.yolo_world:
                # Manual YOLO detection
                h, w = frame.shape[:2]
                if roi:
                    x1, y1 = int(roi[0] * w), int(roi[1] * h)
                    x2, y2 = int(roi[2] * w), int(roi[3] * h)
                    work_frame = frame[y1:y2, x1:x2]
                    offset = (x1, y1)
                else:
                    work_frame = frame
                    offset = (0, 0)
                yolo_results = detector.yolo_world.predict(work_frame, conf=0.25, verbose=False)
                boxes = []
                for r in yolo_results:
                    if r.boxes is None:
                        continue
                    for i in range(len(r.boxes)):
                        bbox = r.boxes.xyxy[i].cpu().numpy().astype(int)
                        boxes.append(DetectedBox(
                            x1=int(bbox[0]) + offset[0],
                            y1=int(bbox[1]) + offset[1],
                            x2=int(bbox[2]) + offset[0],
                            y2=int(bbox[3]) + offset[1],
                            confidence=float(r.boxes.conf[i]),
                            method="yolo_world"
                        ))
            elif current_method == "grounding_dino" and detector.grounding_dino:
                boxes = detector.grounding_dino.detect(frame, "cardboard box", roi)
            else:
                boxes = []

            display = draw_detections(frame, boxes, current_method, roi)

        cv2.imshow(window, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_method = "edge"
            show_all = False
            print(f"Method: Edge Detection")
        elif key == ord('2'):
            current_method = "color"
            show_all = False
            print(f"Method: Color Segmentation")
        elif key == ord('3'):
            current_method = "hybrid"
            show_all = False
            print(f"Method: Hybrid")
        elif key == ord('4'):
            current_method = "yolo_world"
            show_all = False
            print(f"Method: YOLO-World")
        elif key == ord('5'):
            if detector.grounding_dino is None or detector.grounding_dino.model is None:
                print("Loading GroundingDINO (first time takes a while)...")
                detector.load_grounding_dino()
            current_method = "grounding_dino"
            show_all = False
            print(f"Method: GroundingDINO")
        elif key == ord('a'):
            show_all = not show_all
            print(f"Compare all: {show_all}")
        elif key == ord('r'):
            roi = draw_roi(frame)
            print(f"ROI set: {roi}")
        elif key == ord('s'):
            fname = f"detection_{int(time.time())}.jpg"
            cv2.imwrite(fname, display)
            print(f"Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


def draw_detections(frame: np.ndarray, boxes: List[DetectedBox], method: str, roi=None) -> np.ndarray:
    """Draw detections on frame."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Draw ROI
    if roi:
        rx1, ry1 = int(roi[0] * w), int(roi[1] * h)
        rx2, ry2 = int(roi[2] * w), int(roi[3] * h)
        cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

    # Draw boxes
    for i, box in enumerate(boxes):
        cv2.rectangle(display, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
        cv2.putText(display, f"{i+1}", (box.x1+5, box.y1+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Info panel
    cv2.rectangle(display, (10, 10), (300, 70), (0, 0, 0), -1)
    cv2.putText(display, f"Method: {method}", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display, f"Detected: {len(boxes)} boxes", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Instructions
    cv2.rectangle(display, (0, h-30), (w, h), (40, 40, 40), -1)
    cv2.putText(display, "1-5: Methods | A: Compare All | R: ROI | Q: Quit",
               (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return display


def create_comparison_view(frame: np.ndarray, results: dict, roi=None) -> np.ndarray:
    """Create side-by-side comparison of all methods."""
    h, w = frame.shape[:2]

    # Create grid
    methods = list(results.keys())
    cols = min(3, len(methods))
    rows = (len(methods) + cols - 1) // cols

    cell_h = h // rows
    cell_w = w // cols

    grid = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)

    for i, (method, data) in enumerate(results.items()):
        row = i // cols
        col = i % cols

        # Resize frame
        cell = cv2.resize(frame.copy(), (cell_w, cell_h))

        # Draw boxes
        scale_x = cell_w / w
        scale_y = cell_h / h

        for box in data["boxes"]:
            x1 = int(box.x1 * scale_x)
            y1 = int(box.y1 * scale_y)
            x2 = int(box.x2 * scale_x)
            y2 = int(box.y2 * scale_y)
            cv2.rectangle(cell, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Label
        cv2.rectangle(cell, (0, 0), (cell_w, 40), (0, 0, 0), -1)
        cv2.putText(cell, f"{method}: {len(data['boxes'])} ({data['time_ms']:.0f}ms)",
                   (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Place in grid
        y1 = row * cell_h
        y2 = y1 + cell_h
        x1 = col * cell_w
        x2 = x1 + cell_w
        grid[y1:y2, x1:x2] = cell

    return grid


def draw_roi(frame: np.ndarray) -> Tuple[float, float, float, float]:
    """Let user draw ROI."""
    roi_pts = []
    drawing = False
    temp = frame.copy()

    def mouse_cb(event, x, y, flags, param):
        nonlocal roi_pts, drawing, temp
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_pts = [(x, y)]
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp = frame.copy()
            cv2.rectangle(temp, roi_pts[0], (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            roi_pts.append((x, y))
            drawing = False

    win = "Draw ROI - ENTER to confirm"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        cv2.imshow(win, temp)
        key = cv2.waitKey(30) & 0xFF
        if key == 13 and len(roi_pts) == 2:
            break
        elif key == 27:
            cv2.destroyWindow(win)
            return None

    cv2.destroyWindow(win)

    h, w = frame.shape[:2]
    x1 = min(roi_pts[0][0], roi_pts[1][0]) / w
    y1 = min(roi_pts[0][1], roi_pts[1][1]) / h
    x2 = max(roi_pts[0][0], roi_pts[1][0]) / w
    y2 = max(roi_pts[0][1], roi_pts[1][1]) / h

    return (x1, y1, x2, y2)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python box_detector_cv.py <camera_ip>")
        print("Example: python box_detector_cv.py .129")
    else:
        compare_methods(sys.argv[1])
