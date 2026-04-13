#!/usr/bin/env python3
"""
SAM (Segment Anything) Box Counter
===================================

Uses Meta's Segment Anything Model to detect individual boxes
even when they're touching/stacked together.

SAM is excellent at segmenting objects that traditional detection misses.
"""

import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


@dataclass
class SegmentedBox:
    """A segmented box region."""
    x1: int
    y1: int
    x2: int
    y2: int
    mask: np.ndarray
    area: int
    score: float


class SAMBoxCounter:
    """
    Count boxes using Segment Anything Model.
    """

    def __init__(self):
        self.model = None
        self.predictor = None
        self.device = "cpu"
        self.roi = None
        self.last_count = 0
        self.baseline = 0

        # Box size filters
        self.min_area = 1000
        self.max_area = 50000

    def load(self) -> bool:
        """Load SAM model."""
        try:
            # Try FastSAM first (faster)
            try:
                from ultralytics import FastSAM
                print("Loading FastSAM (fast version)...")
                self.model = FastSAM("FastSAM-s.pt")
                self.model_type = "fastsam"
                print("  FastSAM ready!")
                return True
            except:
                pass

            # Try MobileSAM (lightweight)
            try:
                from ultralytics import SAM
                print("Loading MobileSAM...")
                self.model = SAM("mobile_sam.pt")
                self.model_type = "mobilesam"
                print("  MobileSAM ready!")
                return True
            except:
                pass

            # Standard SAM
            try:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                import torch

                print("Loading SAM (this may take a moment)...")

                # Check for model file
                model_path = "sam_vit_b_01ec64.pth"
                if not os.path.exists(model_path):
                    print(f"  Downloading SAM model...")
                    import urllib.request
                    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                    urllib.request.urlretrieve(url, model_path)

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                sam = sam_model_registry["vit_b"](checkpoint=model_path)
                sam.to(device=self.device)

                self.model = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=16,
                    pred_iou_thresh=0.8,
                    stability_score_thresh=0.9,
                    min_mask_region_area=self.min_area,
                )
                self.model_type = "sam"
                print(f"  SAM ready! (using {self.device})")
                return True

            except ImportError:
                print("SAM not installed!")
                print("Install with: pip install segment-anything")
                return False

        except Exception as e:
            print(f"Error loading SAM: {e}")
            return False

    def segment(self, frame: np.ndarray) -> List[SegmentedBox]:
        """Segment boxes in frame."""
        if self.model is None:
            return []

        h, w = frame.shape[:2]

        # Apply ROI
        if self.roi:
            x1 = int(self.roi[0] * w)
            y1 = int(self.roi[1] * h)
            x2 = int(self.roi[2] * w)
            y2 = int(self.roi[3] * h)
            work_frame = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            work_frame = frame
            offset = (0, 0)

        boxes = []

        if self.model_type == "fastsam":
            # FastSAM
            results = self.model(work_frame, retina_masks=True, conf=0.4, iou=0.9)
            if results and len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                for mask in masks:
                    box = self._mask_to_box(mask, offset)
                    if box and self.min_area <= box.area <= self.max_area:
                        boxes.append(box)

        elif self.model_type == "mobilesam":
            # MobileSAM via ultralytics
            results = self.model(work_frame)
            if results and len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                for mask in masks:
                    box = self._mask_to_box(mask, offset)
                    if box and self.min_area <= box.area <= self.max_area:
                        boxes.append(box)

        else:
            # Standard SAM
            rgb = cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB)
            masks = self.model.generate(rgb)

            for mask_data in masks:
                mask = mask_data["segmentation"]
                area = mask_data["area"]
                score = mask_data["predicted_iou"]

                if not (self.min_area <= area <= self.max_area):
                    continue

                # Get bounding box
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    continue

                bx1 = int(xs.min()) + offset[0]
                by1 = int(ys.min()) + offset[1]
                bx2 = int(xs.max()) + offset[0]
                by2 = int(ys.max()) + offset[1]

                boxes.append(SegmentedBox(
                    x1=bx1, y1=by1, x2=bx2, y2=by2,
                    mask=mask, area=area, score=score
                ))

        return boxes

    def _mask_to_box(self, mask: np.ndarray, offset: Tuple[int, int]) -> Optional[SegmentedBox]:
        """Convert mask to bounding box."""
        if mask.ndim == 3:
            mask = mask[0]

        mask_binary = (mask > 0.5).astype(np.uint8)
        ys, xs = np.where(mask_binary)

        if len(xs) == 0 or len(ys) == 0:
            return None

        x1 = int(xs.min()) + offset[0]
        y1 = int(ys.min()) + offset[1]
        x2 = int(xs.max()) + offset[0]
        y2 = int(ys.max()) + offset[1]
        area = int(mask_binary.sum())

        return SegmentedBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            mask=mask_binary, area=area, score=0.9
        )

    def count(self, frame: np.ndarray) -> Tuple[int, List[SegmentedBox]]:
        """Count boxes."""
        boxes = self.segment(frame)
        self.last_count = len(boxes)
        return self.last_count, boxes

    def set_roi_interactive(self, frame: np.ndarray):
        """Draw ROI."""
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
                cv2.rectangle(temp, roi_pts[0], (x, y), (0, 255, 255), 2)
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
                return

        cv2.destroyWindow(win)

        h, w = frame.shape[:2]
        self.roi = (
            min(roi_pts[0][0], roi_pts[1][0]) / w,
            min(roi_pts[0][1], roi_pts[1][1]) / h,
            max(roi_pts[0][0], roi_pts[1][0]) / w,
            max(roi_pts[0][1], roi_pts[1][1]) / h,
        )
        print(f"ROI: {self.roi}")

    def draw_overlay(self, frame: np.ndarray, boxes: List[SegmentedBox]) -> np.ndarray:
        """Draw results."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw ROI
        if self.roi:
            rx1 = int(self.roi[0] * w)
            ry1 = int(self.roi[1] * h)
            rx2 = int(self.roi[2] * w)
            ry2 = int(self.roi[3] * h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

        # Draw boxes with different colors
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        ]

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            cv2.rectangle(display, (box.x1, box.y1), (box.x2, box.y2), color, 2)
            cv2.putText(display, str(i + 1), (box.x1 + 5, box.y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Count panel
        cv2.rectangle(display, (10, 10), (200, 80), (0, 0, 0), -1)
        cv2.putText(display, f"Count: {len(boxes)}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.baseline > 0:
            change = len(boxes) - self.baseline
            color = (0, 255, 0) if change >= 0 else (0, 0, 255)
            cv2.putText(display, f"Change: {change:+d}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return display


def run_sam_counter(camera_ip: str):
    """Run SAM-based counter."""
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2?rtsp_transport=tcp"

    print("=" * 50)
    print("  SAM Box Counter")
    print("=" * 50)

    counter = SAMBoxCounter()
    if not counter.load():
        print("\nFailed to load SAM. Install with:")
        print("  pip install ultralytics  (for FastSAM)")
        print("  or")
        print("  pip install segment-anything torch torchvision")
        return

    print(f"\nConnecting to {camera_ip}...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Failed to connect!")
        return

    print("Connected!")

    # Get frame for ROI
    cap.grab()
    ret, frame = cap.retrieve()
    if ret:
        counter.set_roi_interactive(frame)

    print("\nControls:")
    print("  B - Set baseline")
    print("  R - Redraw ROI")
    print("  S - Screenshot")
    print("  Q - Quit")
    print("-" * 50)

    window = "SAM Box Counter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    boxes = []
    last_detect = 0
    detect_interval = 3.0  # SAM is slower, detect less often

    try:
        while True:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                continue

            # Detect periodically
            if time.time() - last_detect >= detect_interval:
                print("Detecting...", end=" ", flush=True)
                start = time.time()
                count, boxes = counter.count(frame)
                elapsed = time.time() - start
                print(f"{count} boxes ({elapsed:.1f}s)")
                last_detect = time.time()

            display = counter.draw_overlay(frame, boxes)

            h = display.shape[0]
            cv2.rectangle(display, (0, h - 30), (display.shape[1], h), (40, 40, 40), -1)
            cv2.putText(display, "B:Baseline R:ROI S:Screenshot Q:Quit",
                       (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(window, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                counter.baseline = counter.last_count
                print(f"Baseline: {counter.baseline}")
            elif key == ord('r'):
                counter.set_roi_interactive(frame)
            elif key == ord('s'):
                fname = f"sam_{int(time.time())}.jpg"
                cv2.imwrite(fname, display)
                print(f"Saved: {fname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sam_counter.py <camera_ip>")
    else:
        run_sam_counter(sys.argv[1])
