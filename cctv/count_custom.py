#!/usr/bin/env python3
"""
Custom YOLO Box Counter - Trained on YOUR boxes!
With ACTIVE LEARNING - gets smarter over time!
"""

import cv2
import os
import time
import json
from datetime import datetime

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


class CustomCounter:
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path
        self.roi = None
        self.baseline = 0
        self.last_count = 0
        self.confidence = 0.5

        # Stabilization
        self.count_history = []
        self.history_size = 5  # Increased for more stability
        self.stable_count = 0
        self.calibrated = False
        self.calibrated_count = 0

        # Active learning - save training data
        self.training_dir = os.path.join(os.path.dirname(__file__), "active_learning")
        os.makedirs(self.training_dir, exist_ok=True)
        self.samples_collected = len([f for f in os.listdir(self.training_dir) if f.endswith('.jpg')])

        # Multi-scale detection
        self.use_multiscale = True
        self.scales = [1.0, 0.8, 1.2]  # Try different scales

    def load(self):
        """Load trained YOLO model."""
        try:
            from ultralytics import YOLO

            if not os.path.exists(self.model_path):
                print(f"Model not found: {self.model_path}")
                print("Train first with: python train_yolo.py")
                return False

            print(f"Loading custom model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("  Model ready! (YOUR boxes, YOUR accuracy!)")
            return True

        except Exception as e:
            print(f"Error: {e}")
            return False

    def detect(self, frame):
        """Detect boxes with multi-scale for better accuracy."""
        if self.model is None:
            return []

        h, w = frame.shape[:2]

        # Apply ROI
        if self.roi:
            rx1 = int(self.roi[0] * w)
            ry1 = int(self.roi[1] * h)
            rx2 = int(self.roi[2] * w)
            ry2 = int(self.roi[3] * h)
            work = frame[ry1:ry2, rx1:rx2]
            offset = (rx1, ry1)
        else:
            work = frame
            offset = (0, 0)

        all_detections = []

        # Multi-scale detection for better coverage
        scales = self.scales if self.use_multiscale else [1.0]

        for scale in scales:
            if scale != 1.0:
                sh, sw = int(work.shape[0] * scale), int(work.shape[1] * scale)
                if sh < 100 or sw < 100:
                    continue
                scaled = cv2.resize(work, (sw, sh))
            else:
                scaled = work
                scale = 1.0

            # Run YOLO with lower confidence for multi-scale, filter later
            conf = self.confidence if scale == 1.0 else self.confidence * 0.8
            results = self.model(scaled, conf=conf, verbose=False)

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf_score = float(box.conf[0])

                    # Scale back to original size
                    if scale != 1.0:
                        x1, y1 = int(x1 / scale), int(y1 / scale)
                        x2, y2 = int(x2 / scale), int(y2 / scale)

                    all_detections.append({
                        'x1': x1 + offset[0],
                        'y1': y1 + offset[1],
                        'x2': x2 + offset[0],
                        'y2': y2 + offset[1],
                        'score': conf_score
                    })

        # NMS to remove duplicates from multi-scale
        detections = self._nms(all_detections, iou_thresh=0.4)

        # Filter by final confidence
        detections = [d for d in detections if d['score'] >= self.confidence]

        # Stabilize count
        raw_count = len(detections)
        self.count_history.append(raw_count)
        if len(self.count_history) > self.history_size:
            self.count_history.pop(0)

        sorted_counts = sorted(self.count_history)
        median_count = sorted_counts[len(sorted_counts) // 2]

        if self.calibrated:
            diff = abs(median_count - self.calibrated_count)
            if diff >= 2 and len(self.count_history) >= self.history_size:
                if all(abs(c - self.calibrated_count) >= 2 for c in self.count_history[-3:]):
                    self.stable_count = median_count
                    self.calibrated_count = median_count
            else:
                self.stable_count = self.calibrated_count
        else:
            self.stable_count = median_count

        self.last_count = self.stable_count
        self._last_frame = frame.copy()  # Save for training
        self._last_detections = detections
        return detections

    def _nms(self, detections, iou_thresh=0.4):
        """Non-maximum suppression to remove duplicates."""
        if not detections:
            return []

        # Sort by score
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if self._iou(best, d) < iou_thresh
            ]

        return keep

    def _iou(self, a, b):
        """Calculate IoU between two boxes."""
        x1 = max(a['x1'], b['x1'])
        y1 = max(a['y1'], b['y1'])
        x2 = min(a['x2'], b['x2'])
        y2 = min(a['y2'], b['y2'])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area_a = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
        area_b = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])

        return inter / (area_a + area_b - inter + 1e-6)

    def save_training_sample(self, frame, detections):
        """Save frame + detections as YOLO training data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"sample_{timestamp}.jpg"
        label_name = f"sample_{timestamp}.txt"

        img_path = os.path.join(self.training_dir, img_name)
        label_path = os.path.join(self.training_dir, label_name)

        # Save image
        cv2.imwrite(img_path, frame)

        # Save YOLO format labels
        h, w = frame.shape[:2]
        with open(label_path, 'w') as f:
            for d in detections:
                # YOLO format: class x_center y_center width height (normalized)
                cx = ((d['x1'] + d['x2']) / 2) / w
                cy = ((d['y1'] + d['y2']) / 2) / h
                bw = (d['x2'] - d['x1']) / w
                bh = (d['y2'] - d['y1']) / h
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        self.samples_collected += 1
        print(f"Saved training sample! Total: {self.samples_collected}")

    def calibrate(self, detections):
        """Lock count and save as training data."""
        self.calibrated = True
        self.calibrated_count = len(detections)
        self.stable_count = self.calibrated_count

        # Save as training data for active learning
        if hasattr(self, '_last_frame') and self._last_frame is not None:
            self.save_training_sample(self._last_frame, detections)

        print(f"LOCKED to {self.calibrated_count} boxes! (saved for training)")

    def set_roi(self, frame):
        """Draw ROI."""
        h, w = frame.shape[:2]
        max_w, max_h = 1280, 720
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            display = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            display = frame.copy()
            scale = 1.0

        pts = []
        drawing = False
        temp = display.copy()

        def mouse(event, x, y, flags, param):
            nonlocal pts, drawing, temp
            if event == cv2.EVENT_LBUTTONDOWN:
                pts = [(x, y)]
                drawing = True
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp = display.copy()
                cv2.rectangle(temp, pts[0], (x, y), (0, 255, 255), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                pts.append((x, y))
                drawing = False

        win = "Draw ROI - ENTER confirm, ESC skip"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, mouse)

        while True:
            cv2.imshow(win, temp)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and len(pts) == 2:
                break
            elif key == 27:
                cv2.destroyWindow(win)
                return

        cv2.destroyWindow(win)

        dh, dw = display.shape[:2]
        self.roi = (
            min(pts[0][0], pts[1][0]) / dw,
            min(pts[0][1], pts[1][1]) / dh,
            max(pts[0][0], pts[1][0]) / dw,
            max(pts[0][1], pts[1][1]) / dh,
        )
        print("ROI set!")

    def draw(self, frame, detections):
        """Draw results."""
        display = frame.copy()
        h, w = display.shape[:2]

        if self.roi:
            rx1 = int(self.roi[0] * w)
            ry1 = int(self.roi[1] * h)
            rx2 = int(self.roi[2] * w)
            ry2 = int(self.roi[3] * h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

        for i, d in enumerate(detections):
            cv2.rectangle(display, (d['x1'], d['y1']), (d['x2'], d['y2']), (0, 255, 0), 2)
            cv2.putText(display, f"{i+1}", (d['x1'] + 5, d['y1'] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.rectangle(display, (10, 10), (220, 100), (0, 0, 0), -1)

        if self.calibrated:
            cv2.putText(display, "LOCKED", (150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(display, f"Boxes: {self.stable_count}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(display, f"(raw: {len(detections)})", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        if self.baseline > 0:
            change = self.stable_count - self.baseline
            color = (0, 255, 0) if change >= 0 else (0, 0, 255)
            cv2.putText(display, f"Change: {change:+d}", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.rectangle(display, (0, h - 25), (w, h), (40, 40, 40), -1)
        cv2.putText(display, "C:Lock+Save U:Unlock B:Baseline R:ROI M:MultiScale T:Train +/-:Conf Q:Quit",
                   (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv2.putText(display, f"Conf:{self.confidence:.2f}", (w - 100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show training samples count
        cv2.putText(display, f"Samples:{self.samples_collected}", (w - 100, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Show multi-scale status
        ms_status = "MS:ON" if self.use_multiscale else "MS:OFF"
        cv2.putText(display, ms_status, (w - 100, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if self.use_multiscale else (100, 100, 100), 1)

        return display


def run(camera_ip):
    """Run counter."""
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2?rtsp_transport=tcp"

    print("=" * 50)
    print("  CUSTOM YOLO BOX COUNTER")
    print("  (Trained on YOUR boxes!)")
    print("=" * 50)

    model_path = os.path.join(os.path.dirname(__file__), "box_model.pt")
    counter = CustomCounter(model_path)

    if not counter.load():
        return

    print(f"\nConnecting to {camera_ip}...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("No connect!")
        return

    print("Connected!")

    cap.grab()
    ret, frame = cap.retrieve()
    if ret:
        counter.set_roi(frame)

    print("\nControls: C=Lock U=Unlock B=Baseline R=ROI +/-=Confidence Q=Quit")

    win = "Custom YOLO Counter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                continue

            detections = counter.detect(frame)
            display = counter.draw(frame, detections)

            # Resize for display
            dh, dw = display.shape[:2]
            scale = min(1280 / dw, 720 / dh, 1.0)
            if scale < 1.0:
                display = cv2.resize(display, (int(dw * scale), int(dh * scale)))

            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                counter.calibrate(detections)
            elif key == ord('u'):
                counter.calibrated = False
                print("Unlocked!")
            elif key == ord('b'):
                counter.baseline = counter.last_count
                print(f"Baseline: {counter.baseline}")
            elif key == ord('r'):
                counter.set_roi(frame)
            elif key == ord('+') or key == ord('='):
                counter.confidence = min(0.9, counter.confidence + 0.05)
                print(f"Confidence: {counter.confidence:.2f}")
            elif key == ord('-'):
                counter.confidence = max(0.1, counter.confidence - 0.05)
                print(f"Confidence: {counter.confidence:.2f}")
            elif key == ord('m'):
                counter.use_multiscale = not counter.use_multiscale
                print(f"Multi-scale: {'ON' if counter.use_multiscale else 'OFF'}")
            elif key == ord('t'):
                print(f"\nRetraining with {counter.samples_collected} samples...")
                retrain_model(counter)
            elif key == ord('s'):
                # Manual save current frame
                if hasattr(counter, '_last_frame'):
                    counter.save_training_sample(counter._last_frame, detections)

    finally:
        cap.release()
        cv2.destroyAllWindows()


def retrain_model(counter):
    """Quick retrain with active learning samples."""

    training_dir = counter.training_dir
    samples = [f for f in os.listdir(training_dir) if f.endswith('.jpg')]

    if len(samples) < 5:
        print(f"Need at least 5 samples to retrain. Have: {len(samples)}")
        return

    print(f"Found {len(samples)} samples. Starting quick retrain...")

    # Create dataset YAML
    yaml_content = f"""
path: {training_dir}
train: .
val: .

names:
  0: box
"""
    yaml_path = os.path.join(training_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # Run training (quick - few epochs)
    try:
        from ultralytics import YOLO

        # Start from current model for transfer learning
        model = YOLO(counter.model_path)

        print("Training... (this may take a few minutes)")
        model.train(
            data=yaml_path,
            epochs=10,  # Quick retrain
            imgsz=640,
            batch=4,
            patience=3,
            project=training_dir,
            name="retrain",
            exist_ok=True
        )

        # Update model path
        new_model = os.path.join(training_dir, "retrain", "weights", "best.pt")
        if os.path.exists(new_model):
            counter.model = YOLO(new_model)
            print(f"Model updated! Now smarter with {len(samples)} samples.")
        else:
            print("Training completed but no improvement found.")

    except Exception as e:
        print(f"Retrain error: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Custom YOLO Box Counter")
        print("=" * 30)
        print("\nUsage: python count_custom.py <camera_ip>")
        print("Example: python count_custom.py .129")
    else:
        run(sys.argv[1])
