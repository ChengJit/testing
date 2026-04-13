#!/usr/bin/env python3
"""
Terminal Box Counter
====================

Specialized counter for terminal boxes on shelf.
Uses zone filtering + size filtering to count only terminal boxes.

Features:
- Define ROI (region of interest) for terminal area
- Filter by box size to ignore large boxes
- Track count changes
- Alert when boxes removed
"""

import cv2
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


@dataclass
class BoxDetection:
    """Detected box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    area: int

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class TerminalCounter:
    """
    Counts terminal boxes in a defined zone.
    """

    def __init__(
        self,
        # ROI for terminal box area (normalized 0-1)
        roi: Tuple[float, float, float, float] = (0.15, 0.35, 0.95, 0.95),
        # Box size range (pixels) - adjust based on your camera
        min_box_area: int = 1000,
        max_box_area: int = 15000,
        min_box_width: int = 30,
        max_box_width: int = 150,
        # Detection threshold
        conf_threshold: float = 0.25,
    ):
        self.roi = roi  # x1, y1, x2, y2 (normalized)
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.min_box_width = min_box_width
        self.max_box_width = max_box_width
        self.conf_threshold = conf_threshold

        # Detector
        self.model = None
        self._load_detector()

        # State
        self.last_count = 0
        self.baseline_count = 0
        self.history: List[Tuple[datetime, int]] = []

    def _load_detector(self):
        """Load YOLO-World detector."""
        try:
            from ultralytics import YOLO

            print("Loading YOLO-World for box detection...")
            self.model = YOLO("yolov8s-worldv2.pt")
            self.model.set_classes(["cardboard box", "box", "carton"])
            print("  Detector ready")

        except Exception as e:
            print(f"  Detector error: {e}")
            self.model = None

    def set_roi_interactive(self, frame: np.ndarray) -> Tuple[float, float, float, float]:
        """Let user draw ROI on frame."""
        print("\nDraw ROI for terminal box area:")
        print("  - Click and drag to draw rectangle")
        print("  - Press ENTER to confirm")
        print("  - Press R to reset")
        print("  - Press C to cancel (use default)")

        roi_points = []
        drawing = False
        temp_frame = frame.copy()

        def mouse_callback(event, x, y, flags, param):
            nonlocal roi_points, drawing, temp_frame

            if event == cv2.EVENT_LBUTTONDOWN:
                roi_points = [(x, y)]
                drawing = True

            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp_frame = frame.copy()
                cv2.rectangle(temp_frame, roi_points[0], (x, y), (0, 255, 0), 2)

            elif event == cv2.EVENT_LBUTTONUP:
                roi_points.append((x, y))
                drawing = False

        window = "Draw ROI - ENTER to confirm, R to reset"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse_callback)

        while True:
            display = temp_frame.copy()

            # Draw current ROI
            if len(roi_points) == 2:
                cv2.rectangle(display, roi_points[0], roi_points[1], (0, 255, 0), 2)
                cv2.putText(display, "Press ENTER to confirm", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 13 and len(roi_points) == 2:  # ENTER
                break
            elif key == ord('r'):  # Reset
                roi_points = []
                temp_frame = frame.copy()
            elif key == ord('c') or key == 27:  # Cancel/ESC
                cv2.destroyWindow(window)
                return self.roi

        cv2.destroyWindow(window)

        # Convert to normalized coordinates
        h, w = frame.shape[:2]
        x1 = min(roi_points[0][0], roi_points[1][0]) / w
        y1 = min(roi_points[0][1], roi_points[1][1]) / h
        x2 = max(roi_points[0][0], roi_points[1][0]) / w
        y2 = max(roi_points[0][1], roi_points[1][1]) / h

        self.roi = (x1, y1, x2, y2)
        print(f"ROI set: {self.roi}")
        return self.roi

    def detect_boxes(self, frame: np.ndarray) -> List[BoxDetection]:
        """Detect terminal boxes in frame."""
        if self.model is None:
            return []

        h, w = frame.shape[:2]

        # Crop to ROI
        x1 = int(self.roi[0] * w)
        y1 = int(self.roi[1] * h)
        x2 = int(self.roi[2] * w)
        y2 = int(self.roi[3] * h)
        roi_frame = frame[y1:y2, x1:x2]

        # Detect
        results = self.model.predict(roi_frame, conf=self.conf_threshold, verbose=False)

        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                bbox = r.boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(r.boxes.conf[i])

                # Convert back to full frame coordinates
                bx1 = int(bbox[0]) + x1
                by1 = int(bbox[1]) + y1
                bx2 = int(bbox[2]) + x1
                by2 = int(bbox[3]) + y1

                box_w = bx2 - bx1
                box_h = by2 - by1
                area = box_w * box_h

                # Filter by size (terminal boxes only)
                if (self.min_box_area <= area <= self.max_box_area and
                    self.min_box_width <= box_w <= self.max_box_width):
                    boxes.append(BoxDetection(
                        x1=bx1, y1=by1, x2=bx2, y2=by2,
                        confidence=conf, area=area
                    ))

        return boxes

    def count(self, frame: np.ndarray) -> Tuple[int, List[BoxDetection]]:
        """Count terminal boxes and return detections."""
        boxes = self.detect_boxes(frame)
        count = len(boxes)

        # Track change
        change = count - self.last_count
        self.last_count = count
        self.history.append((datetime.now(), count))

        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return count, boxes

    def set_baseline(self, count: int):
        """Set baseline count for comparison."""
        self.baseline_count = count
        print(f"Baseline set: {count} boxes")

    def get_change_from_baseline(self) -> int:
        """Get change from baseline."""
        return self.last_count - self.baseline_count

    def draw_overlay(self, frame: np.ndarray, boxes: List[BoxDetection]) -> np.ndarray:
        """Draw detection overlay on frame."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw ROI
        rx1 = int(self.roi[0] * w)
        ry1 = int(self.roi[1] * h)
        rx2 = int(self.roi[2] * w)
        ry2 = int(self.roi[3] * h)
        cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
        cv2.putText(display, "Terminal Zone", (rx1 + 5, ry1 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw boxes
        for i, box in enumerate(boxes):
            cv2.rectangle(display, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
            cv2.putText(display, f"{i+1}", (box.x1 + 5, box.y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Count display
        count = len(boxes)
        change = self.get_change_from_baseline()

        cv2.rectangle(display, (10, 10), (200, 80), (0, 0, 0), -1)
        cv2.putText(display, f"Count: {count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.baseline_count > 0:
            change_color = (0, 255, 0) if change >= 0 else (0, 0, 255)
            change_text = f"Change: {change:+d}"
            cv2.putText(display, change_text, (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, change_color, 2)

        return display


def run_terminal_counter(camera_ip: str):
    """Run terminal counter on camera."""

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

    print("Connected!")

    # Create counter
    counter = TerminalCounter()

    # Get first frame for ROI setup
    cap.grab()
    ret, frame = cap.retrieve()
    if ret:
        print("\nSetup ROI for terminal box area...")
        counter.set_roi_interactive(frame)

    print("\nControls:")
    print("  B - Set current count as BASELINE")
    print("  R - Redraw ROI")
    print("  S - Screenshot")
    print("  Q - Quit")
    print("-" * 40)

    window = "Terminal Counter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last_detect = 0
    detect_interval = 2.0  # seconds

    try:
        while True:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                continue

            # Detect periodically
            if time.time() - last_detect >= detect_interval:
                count, boxes = counter.count(frame)
                last_detect = time.time()

                # Log changes
                if counter.baseline_count > 0:
                    change = counter.get_change_from_baseline()
                    if change != 0:
                        print(f"[{datetime.now():%H:%M:%S}] Count: {count} (change: {change:+d})")

            # Draw overlay
            display = counter.draw_overlay(frame, boxes if 'boxes' in dir() else [])

            # Instructions
            h = display.shape[0]
            cv2.rectangle(display, (0, h-30), (display.shape[1], h), (40, 40, 40), -1)
            cv2.putText(display, "B:Baseline | R:ROI | S:Screenshot | Q:Quit",
                       (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(window, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                counter.set_baseline(counter.last_count)
            elif key == ord('r'):
                counter.set_roi_interactive(frame)
            elif key == ord('s'):
                fname = f"terminal_{int(time.time())}.jpg"
                cv2.imwrite(fname, display)
                print(f"Saved: {fname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python terminal_counter.py <camera_ip>")
        print("Example: python terminal_counter.py .129")
    else:
        run_terminal_counter(sys.argv[1])
