#!/usr/bin/env python3
"""
OWLv2 Box Counter
==================

Google's OWL-ViT v2 - Open-Vocabulary Object Detection
Very accurate for finding specific objects by text prompt.

More stable than Florence-2 and works well for box detection.
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
class Detection:
    """Detected object."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    score: float

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class OWLv2Counter:
    """
    Count boxes using Google's OWLv2 model.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.roi = None
        self.last_count = 0
        self.baseline = 0
        self.queries = ["cardboard box", "box", "package"]
        self.min_area = 500
        self.max_area = 50000
        self.threshold = 0.1

    def load(self) -> bool:
        """Load OWLv2 model."""
        try:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            import torch

            print("Loading OWLv2 (Google's open-vocabulary detector)...")

            # Check GPU
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"  Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                print("  Using CPU")

            model_id = "google/owlv2-base-patch16-ensemble"

            self.processor = Owlv2Processor.from_pretrained(model_id)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)

            print("  OWLv2 ready!")
            return True

        except ImportError:
            print("OWLv2 requires: pip install transformers torch")
            return False
        except Exception as e:
            print(f"Error loading OWLv2: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect boxes in frame."""
        if self.model is None:
            return []

        import torch
        from PIL import Image

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

        # Resize for faster processing (OWLv2 is slow on large images)
        work_h, work_w = work_frame.shape[:2]
        max_size = 512  # Smaller = faster
        scale = min(max_size / work_w, max_size / work_h, 1.0)
        if scale < 1.0:
            new_w = int(work_w * scale)
            new_h = int(work_h * scale)
            work_frame_resized = cv2.resize(work_frame, (new_w, new_h))
        else:
            work_frame_resized = work_frame
            scale = 1.0

        # Convert to PIL
        image = Image.fromarray(cv2.cvtColor(work_frame_resized, cv2.COLOR_BGR2RGB))

        # Process
        inputs = self.processor(
            text=[self.queries],
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process - extract boxes manually from outputs
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)

        # Get predictions
        logits = outputs.logits[0]  # [num_queries, num_classes]
        boxes = outputs.pred_boxes[0]  # [num_queries, 4]

        # Get scores and filter by threshold
        probs = torch.sigmoid(logits)
        scores, labels = probs.max(dim=-1)

        # Filter by threshold
        mask = scores > self.threshold
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]
        filtered_boxes = boxes[mask]

        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        img_h, img_w = image.size[::-1]
        cx, cy, bw, bh = filtered_boxes.unbind(-1)
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h

        converted_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        results = {
            "boxes": converted_boxes,
            "scores": filtered_scores,
            "labels": filtered_labels
        }

        detections = []
        for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
            bx1, by1, bx2, by2 = box.cpu().numpy()

            # Scale back to original size
            bx1 = int(bx1 / scale) + offset[0]
            by1 = int(by1 / scale) + offset[1]
            bx2 = int(bx2 / scale) + offset[0]
            by2 = int(by2 / scale) + offset[1]

            area = (bx2 - bx1) * (by2 - by1)

            if self.min_area <= area <= self.max_area:
                label_i = int(label_idx.cpu().item()) if hasattr(label_idx, 'cpu') else int(label_idx)
                label_i = min(label_i, len(self.queries) - 1)  # Safety clamp
                detections.append(Detection(
                    x1=bx1, y1=by1, x2=bx2, y2=by2,
                    label=self.queries[label_i],
                    score=float(score.cpu().item()) if hasattr(score, 'cpu') else float(score)
                ))

        # Remove duplicates (overlapping detections)
        detections = self._nms(detections, iou_thresh=0.5)

        return detections

    def _nms(self, detections: List[Detection], iou_thresh: float = 0.5) -> List[Detection]:
        """Non-maximum suppression."""
        if not detections:
            return []

        # Sort by score
        detections = sorted(detections, key=lambda d: d.score, reverse=True)

        kept = []
        for det in detections:
            is_duplicate = False
            for existing in kept:
                iou = self._iou(det, existing)
                if iou > iou_thresh:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(det)

        return kept

    def _iou(self, d1: Detection, d2: Detection) -> float:
        """Calculate IoU."""
        x1 = max(d1.x1, d2.x1)
        y1 = max(d1.y1, d2.y1)
        x2 = min(d1.x2, d2.x2)
        y2 = min(d1.y2, d2.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = d1.area + d2.area - intersection
        return intersection / union if union > 0 else 0.0

    def count(self, frame: np.ndarray) -> Tuple[int, List[Detection]]:
        """Count boxes."""
        detections = self.detect(frame)
        self.last_count = len(detections)
        return self.last_count, detections

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

    def draw_overlay(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detections."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw ROI
        if self.roi:
            rx1 = int(self.roi[0] * w)
            ry1 = int(self.roi[1] * h)
            rx2 = int(self.roi[2] * w)
            ry2 = int(self.roi[3] * h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

        # Draw detections
        for i, det in enumerate(detections):
            color = (0, 255, 0)
            cv2.rectangle(display, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            cv2.putText(display, f"{i+1} ({det.score:.0%})", (det.x1 + 5, det.y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Count panel
        cv2.rectangle(display, (10, 10), (200, 80), (0, 0, 0), -1)
        cv2.putText(display, f"Boxes: {len(detections)}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.baseline > 0:
            change = len(detections) - self.baseline
            color = (0, 255, 0) if change >= 0 else (0, 0, 255)
            cv2.putText(display, f"Change: {change:+d}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return display


def run_owl_counter(camera_ip: str):
    """Run OWLv2 counter."""
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2?rtsp_transport=tcp"

    print("=" * 50)
    print("  OWLv2 Box Counter (Google)")
    print("=" * 50)

    counter = OWLv2Counter()
    if not counter.load():
        print("\nInstall: pip install transformers torch")
        return

    print(f"\nConnecting to {camera_ip}...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Failed to connect!")
        return

    print("Connected!")

    cap.grab()
    ret, frame = cap.retrieve()
    if ret:
        counter.set_roi_interactive(frame)

    print("\nControls:")
    print("  B - Set baseline")
    print("  R - Redraw ROI")
    print("  +/- - Adjust threshold")
    print("  S - Screenshot")
    print("  Q - Quit")
    print("-" * 50)

    window = "OWLv2 Box Counter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    import threading

    detections = []
    detecting = False
    detect_frame = None
    detect_result = None
    detect_interval = 3.0  # Longer interval for heavy model

    def detect_thread():
        nonlocal detect_result, detecting
        if detect_frame is not None:
            print(f"Detecting (threshold={counter.threshold:.2f})...", end=" ", flush=True)
            start = time.time()
            count, dets = counter.count(detect_frame)
            elapsed = time.time() - start
            print(f"{count} boxes ({elapsed:.1f}s)")
            detect_result = dets
        detecting = False

    last_detect = 0

    try:
        while True:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                continue

            # Start detection in background thread
            if not detecting and time.time() - last_detect >= detect_interval:
                detect_frame = frame.copy()
                detecting = True
                last_detect = time.time()
                threading.Thread(target=detect_thread, daemon=True).start()

            # Update detections when ready
            if detect_result is not None:
                detections = detect_result
                detect_result = None

            display = counter.draw_overlay(frame, detections)

            # Threshold display
            cv2.putText(display, f"Thresh: {counter.threshold:.2f}", (display.shape[1] - 130, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            h = display.shape[0]
            cv2.rectangle(display, (0, h - 30), (display.shape[1], h), (40, 40, 40), -1)
            cv2.putText(display, "B:Baseline R:ROI +/-:Threshold S:Screenshot Q:Quit",
                       (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow(window, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                counter.baseline = counter.last_count
                print(f"Baseline: {counter.baseline}")
            elif key == ord('r'):
                counter.set_roi_interactive(frame)
            elif key == ord('+') or key == ord('='):
                counter.threshold = min(0.5, counter.threshold + 0.02)
                print(f"Threshold: {counter.threshold:.2f}")
            elif key == ord('-'):
                counter.threshold = max(0.02, counter.threshold - 0.02)
                print(f"Threshold: {counter.threshold:.2f}")
            elif key == ord('s'):
                fname = f"owl_{int(time.time())}.jpg"
                cv2.imwrite(fname, display)
                print(f"Saved: {fname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python owl_counter.py <camera_ip>")
    else:
        run_owl_counter(sys.argv[1])
