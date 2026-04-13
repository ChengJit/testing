#!/usr/bin/env python3
"""
Training Data Collector for Rack Inventory

Workflow:
1. Capture empty rack as baseline
2. Add items to rack
3. Capture frames and annotate
4. Export to YOLO format for training

Usage:
    python -m rack_inventory.training_collector --camera .128
"""

import cv2
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import shutil


@dataclass
class BoundingBox:
    """Annotation bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = "box"

    def to_yolo(self, img_w: int, img_h: int) -> str:
        """Convert to YOLO format: class_id cx cy w h (normalized)."""
        cx = ((self.x1 + self.x2) / 2) / img_w
        cy = ((self.y1 + self.y2) / 2) / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


@dataclass
class TrainingImage:
    """Single training image with annotations."""
    path: str
    boxes: List[BoundingBox] = field(default_factory=list)
    is_baseline: bool = False


class TrainingCollector:
    """
    Collects and annotates training data for rack inventory.
    """

    def __init__(self, output_dir: str = "rack_training_data"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.baseline_dir = self.output_dir / "baseline"

        # Create directories
        for d in [self.images_dir, self.labels_dir, self.baseline_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # State
        self.current_frame: Optional[np.ndarray] = None
        self.current_boxes: List[BoundingBox] = []
        self.drawing = False
        self.start_point = (0, 0)
        self.class_labels = ["box", "terminal", "device", "package"]
        self.current_class = 0

        # Stats
        self.image_count = len(list(self.images_dir.glob("*.jpg")))
        self.baseline_count = len(list(self.baseline_dir.glob("*.jpg")))

    def capture_from_camera(self, camera_url: str):
        """Capture frames from camera for annotation."""
        print("=" * 50)
        print("  Training Data Collector")
        print("=" * 50)
        print()
        print("Controls:")
        print("  B - Capture BASELINE (empty rack)")
        print("  C - Capture frame for annotation")
        print("  Left-click + drag - Draw box")
        print("  Right-click - Remove last box")
        print("  1-4 - Select class (box/terminal/device/package)")
        print("  S - Save current annotations")
        print("  A - Auto-detect boxes (YOLO-World)")
        print("  Q - Quit")
        print("-" * 50)
        print(f"Output: {self.output_dir}")
        print(f"Existing: {self.image_count} images, {self.baseline_count} baselines")
        print()

        cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"ERROR: Cannot connect to {camera_url}")
            return

        window_name = "Training Collector"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        captured_frame = None
        mode = "live"  # live, annotate

        try:
            while True:
                if mode == "live":
                    cap.grab()
                    ret, frame = cap.retrieve()
                    if not ret:
                        continue

                    self.current_frame = frame.copy()
                    display = frame.copy()

                    # Show live indicator
                    cv2.circle(display, (20, 20), 8, (0, 0, 255), -1)
                    cv2.putText(display, "LIVE", (35, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                else:  # annotate mode
                    display = captured_frame.copy()

                    # Draw existing boxes
                    for box in self.current_boxes:
                        color = self._get_class_color(box.label)
                        cv2.rectangle(display, (box.x1, box.y1), (box.x2, box.y2), color, 2)
                        cv2.putText(display, box.label, (box.x1, box.y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Draw current box being drawn
                    if self.drawing:
                        cv2.rectangle(display, self.start_point, self._current_mouse, (255, 255, 0), 2)

                    # Annotation indicator
                    cv2.circle(display, (20, 20), 8, (0, 255, 0), -1)
                    cv2.putText(display, f"ANNOTATE ({len(self.current_boxes)} boxes)", (35, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Class selector
                h = display.shape[0]
                cv2.rectangle(display, (0, h - 30), (display.shape[1], h), (40, 40, 40), -1)
                class_text = f"Class: {self.class_labels[self.current_class]} (1-4 to change) | B:Baseline C:Capture S:Save Q:Quit"
                cv2.putText(display, class_text, (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('b'):
                    # Capture baseline
                    self._save_baseline(self.current_frame)

                elif key == ord('c'):
                    # Capture for annotation
                    captured_frame = self.current_frame.copy()
                    self.current_boxes = []
                    mode = "annotate"
                    print("Frame captured - draw boxes and press S to save")

                elif key == ord('s') and mode == "annotate":
                    # Save with annotations
                    self._save_annotated(captured_frame, self.current_boxes)
                    mode = "live"
                    self.current_boxes = []

                elif key == ord('a') and mode == "annotate":
                    # Auto-detect
                    self.current_boxes = self._auto_detect(captured_frame)
                    print(f"Auto-detected {len(self.current_boxes)} boxes")

                elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                    self.current_class = key - ord('1')
                    print(f"Class: {self.class_labels[self.current_class]}")

                elif key == 27:  # ESC - cancel annotation
                    mode = "live"
                    self.current_boxes = []

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes."""
        self._current_mouse = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = x, y

                # Ensure correct order
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Minimum size check
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self.current_boxes.append(BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        label=self.class_labels[self.current_class]
                    ))
                    print(f"Added box: {self.class_labels[self.current_class]}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last box
            if self.current_boxes:
                self.current_boxes.pop()
                print("Removed last box")

    def _get_class_color(self, label: str) -> Tuple[int, int, int]:
        """Get color for class label."""
        colors = {
            "box": (0, 255, 0),
            "terminal": (255, 0, 0),
            "device": (0, 0, 255),
            "package": (255, 165, 0),
        }
        return colors.get(label, (255, 255, 0))

    def _save_baseline(self, frame: np.ndarray):
        """Save baseline (empty rack) image."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"baseline_{timestamp}.jpg"
        path = self.baseline_dir / filename
        cv2.imwrite(str(path), frame)
        self.baseline_count += 1
        print(f"Saved baseline: {filename} (total: {self.baseline_count})")

    def _save_annotated(self, frame: np.ndarray, boxes: List[BoundingBox]):
        """Save annotated image and labels."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"rack_{timestamp}.jpg"
        label_filename = f"rack_{timestamp}.txt"

        # Save image
        img_path = self.images_dir / img_filename
        cv2.imwrite(str(img_path), frame)

        # Save labels in YOLO format
        h, w = frame.shape[:2]
        label_path = self.labels_dir / label_filename
        with open(label_path, "w") as f:
            for box in boxes:
                f.write(box.to_yolo(w, h) + "\n")

        self.image_count += 1
        print(f"Saved: {img_filename} with {len(boxes)} boxes (total: {self.image_count})")

    def _auto_detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Auto-detect boxes using YOLO-World."""
        try:
            from ultralytics import YOLO

            model = YOLO("yolov8s-worldv2.pt")
            model.set_classes(["cardboard box", "box", "package"])

            results = model.predict(frame, conf=0.3, verbose=False)

            boxes = []
            for r in results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    bbox = r.boxes.xyxy[i].cpu().numpy().astype(int)
                    boxes.append(BoundingBox(
                        x1=int(bbox[0]), y1=int(bbox[1]),
                        x2=int(bbox[2]), y2=int(bbox[3]),
                        label="box"
                    ))
            return boxes

        except Exception as e:
            print(f"Auto-detect error: {e}")
            return []

    def export_yolo_dataset(self, train_split: float = 0.8):
        """Export collected data as YOLO dataset."""
        dataset_dir = self.output_dir / "yolo_dataset"
        train_images = dataset_dir / "train" / "images"
        train_labels = dataset_dir / "train" / "labels"
        val_images = dataset_dir / "val" / "images"
        val_labels = dataset_dir / "val" / "labels"

        for d in [train_images, train_labels, val_images, val_labels]:
            d.mkdir(parents=True, exist_ok=True)

        # Get all images with labels
        images = list(self.images_dir.glob("*.jpg"))

        if not images:
            print("No training images found!")
            return

        # Split train/val
        import random
        random.shuffle(images)
        split_idx = int(len(images) * train_split)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Copy files
        for img_path in train_imgs:
            label_path = self.labels_dir / img_path.name.replace(".jpg", ".txt")
            shutil.copy(img_path, train_images)
            if label_path.exists():
                shutil.copy(label_path, train_labels)

        for img_path in val_imgs:
            label_path = self.labels_dir / img_path.name.replace(".jpg", ".txt")
            shutil.copy(img_path, val_images)
            if label_path.exists():
                shutil.copy(label_path, val_labels)

        # Create dataset.yaml
        yaml_content = f"""
path: {dataset_dir.absolute()}
train: train/images
val: val/images

names:
  0: box
  1: terminal
  2: device
  3: package
"""

        yaml_path = dataset_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(f"\nDataset exported to: {dataset_dir}")
        print(f"  Train: {len(train_imgs)} images")
        print(f"  Val: {len(val_imgs)} images")
        print(f"\nTo train:")
        print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=50")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Training Data Collector")
    parser.add_argument("--camera", "-cam", required=True, help="Camera IP or RTSP URL")
    parser.add_argument("--output", "-o", default="rack_training_data", help="Output directory")
    parser.add_argument("--export", action="store_true", help="Export YOLO dataset")

    args = parser.parse_args()

    collector = TrainingCollector(output_dir=args.output)

    if args.export:
        collector.export_yolo_dataset()
        return

    # Build RTSP URL if needed
    camera_url = args.camera
    if not camera_url.startswith("rtsp://"):
        if camera_url.startswith("."):
            camera_url = f"192.168.122{camera_url}"
        camera_url = f"rtsp://fasspay:fasspay2025@{camera_url}:554/stream2?rtsp_transport=tcp"

    collector.capture_from_camera(camera_url)


if __name__ == "__main__":
    main()
