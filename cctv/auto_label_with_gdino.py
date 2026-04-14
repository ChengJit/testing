#!/usr/bin/env python3
"""
Auto-Labeler using GroundingDINO
Automatically detects boxes and creates YOLO training dataset.

This uses GroundingDINO (slow but accurate) to auto-generate labels,
then you train YOLOv8 (fast) on the labeled data.

Usage:
    python auto_label_with_gdino.py <camera_ip>

Controls:
    SPACE  - Force capture current frame
    A      - Toggle auto-capture mode
    +/-    - Adjust confidence threshold
    Q      - Quit and show summary
"""

import cv2
import numpy as np
import os
import sys
import time
import threading
from datetime import datetime
from PIL import Image

# ============ CONFIGURATION ============
CONFIG = {
    # Camera
    "rtsp_user": "fasspay",
    "rtsp_pass": "fasspay2025",
    "use_hd": True,

    # GroundingDINO settings
    "confidence": 0.25,          # Detection threshold
    "prompt": "cardboard box . brown box . package . box",
    "max_detect_dim": 800,

    # Dataset settings
    "output_dir": "dataset",
    "target_images": 100,
    "min_boxes_per_frame": 1,    # Only save frames with at least this many boxes
    "auto_interval": 2.0,        # Seconds between auto-captures

    # Filtering
    "min_area": 500,
    "max_area": 500000,

    # Image size for training
    "save_width": 1280,
    "save_height": 720,
}
# =======================================


class FrameGrabber:
    def __init__(self, url):
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        gst = (
            f"rtspsrc location={url} latency=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )
        self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return (True, self.frame.copy()) if self.frame is not None else (False, None)

    def release(self):
        self.running = False
        self.cap.release()


class GroundingDINOLabeler:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = "cpu"

    def load(self):
        """Load GroundingDINO model."""
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Device: {self.device}")

            # Find GroundingDINO
            gdino_paths = [
                os.path.expanduser("~/test/GroundingDINO"),
                os.path.expanduser("~/GroundingDINO"),
                "./GroundingDINO",
                "../GroundingDINO",
            ]

            gdino_path = None
            for path in gdino_paths:
                if os.path.exists(path):
                    gdino_path = path
                    break

            if not gdino_path:
                print("ERROR: GroundingDINO not found!")
                return False

            print(f"  GroundingDINO: {gdino_path}")
            sys.path.insert(0, gdino_path)

            from groundingdino.util.inference import load_model

            config_path = os.path.join(gdino_path, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
            weights_path = os.path.join(gdino_path, "weights", "groundingdino_swint_ogc.pth")

            print("  Loading model...")
            self.model = load_model(config_path, weights_path, device=self.device)
            print("  GroundingDINO ready!")
            return True

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def detect(self, frame, confidence):
        """Detect boxes and return YOLO format labels."""
        if self.model is None:
            return [], frame

        import torch
        import groundingdino.datasets.transforms as T
        from groundingdino.util.inference import predict

        h, w = frame.shape[:2]

        # Downscale for speed
        max_dim = self.config["max_detect_dim"]
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            work = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            work = frame

        transform = T.Compose([
            T.RandomResize([640], max_size=800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        image_pil = Image.fromarray(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))
        image_tensor, _ = transform(image_pil, None)

        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=self.config["prompt"],
                box_threshold=confidence,
                text_threshold=confidence,
                device=self.device
            )

        detections = []

        for box, score in zip(boxes, logits):
            cx, cy, bw, bh = box.tolist()

            # GroundingDINO returns normalized coords (0-1)
            # Clamp to valid range to ensure YOLO format compliance
            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            bw = max(0.001, min(0.998, bw))
            bh = max(0.001, min(0.998, bh))

            # Ensure box doesn't exceed image bounds
            if cx - bw/2 < 0:
                bw = cx * 2
            if cx + bw/2 > 1:
                bw = (1 - cx) * 2
            if cy - bh/2 < 0:
                bh = cy * 2
            if cy + bh/2 > 1:
                bh = (1 - cy) * 2

            # Convert to pixel coords for filtering (use original frame size)
            px1 = int((cx - bw/2) * w)
            py1 = int((cy - bh/2) * h)
            px2 = int((cx + bw/2) * w)
            py2 = int((cy + bh/2) * h)

            area = (px2 - px1) * (py2 - py1)
            if self.config["min_area"] <= area <= self.config["max_area"]:
                # YOLO format: class cx cy w h (normalized 0-1)
                detections.append({
                    'yolo': f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}",
                    'pixel': (px1, py1, px2, py2),
                    'score': float(score)
                })

        return detections, frame


def run(camera_ip):
    print("=" * 50)
    print("  AUTO-LABELER (GroundingDINO -> YOLOv8)")
    print("=" * 50)

    # Create directories
    images_dir = os.path.join(CONFIG["output_dir"], "images")
    labels_dir = os.path.join(CONFIG["output_dir"], "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    existing = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    print(f"\nOutput: {CONFIG['output_dir']}/")
    print(f"Existing: {existing} images")
    print(f"Target: {CONFIG['target_images']} images")

    # Load detector
    print("\nLoading GroundingDINO...")
    labeler = GroundingDINOLabeler(CONFIG)
    if not labeler.load():
        return

    # Connect camera
    stream = "stream1" if CONFIG["use_hd"] else "stream2"
    url = f"rtsp://{CONFIG['rtsp_user']}:{CONFIG['rtsp_pass']}@{camera_ip}:554/{stream}"

    print(f"\nConnecting to camera...")
    cap = FrameGrabber(url)
    time.sleep(1)

    ret, frame = cap.read()
    if not ret:
        print("Connection failed!")
        return

    print(f"Connected! {frame.shape[1]}x{frame.shape[0]}")

    print("\n" + "=" * 50)
    print("  CONTROLS")
    print("=" * 50)
    print("  SPACE  - Force capture")
    print("  A      - Toggle auto-mode")
    print("  +/-    - Adjust threshold")
    print("  Q      - Quit")
    print("=" * 50 + "\n")

    cv2.namedWindow("Auto-Labeler", cv2.WINDOW_NORMAL)

    captured = existing
    auto_mode = False
    confidence = CONFIG["confidence"]
    auto_interval = CONFIG["auto_interval"]
    last_auto = 0
    last_capture = 0
    last_detect = 0
    detect_interval = 1.0

    detections = []
    detecting = False
    target = CONFIG["target_images"]
    save_w, save_h = CONFIG["save_width"], CONFIG["save_height"]

    def save_sample(frame, dets):
        nonlocal captured, last_capture

        if len(dets) < CONFIG["min_boxes_per_frame"]:
            return False

        # Resize frame
        resized = cv2.resize(frame, (save_w, save_h))

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"box_{timestamp}_{captured:04d}"

        # Save image
        img_path = os.path.join(images_dir, f"{base_name}.jpg")
        cv2.imwrite(img_path, resized)

        # Save YOLO labels
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        with open(label_path, "w") as f:
            for det in dets:
                f.write(det['yolo'] + "\n")

        captured += 1
        last_capture = time.time()
        print(f"  [SAVED] {base_name} - {len(dets)} boxes ({captured}/{target})")
        return True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Run detection periodically
            if not detecting and time.time() - last_detect >= detect_interval:
                detecting = True
                last_detect = time.time()

                start = time.time()
                detections, _ = labeler.detect(frame, confidence)
                elapsed = time.time() - start
                print(f"  Detected: {len(detections)} boxes ({elapsed:.2f}s)")

                detecting = False

            # Auto-save
            if auto_mode and time.time() - last_auto >= auto_interval:
                if save_sample(frame, detections):
                    last_auto = time.time()
                else:
                    last_auto = time.time() - auto_interval + 1  # Retry soon

            # Draw display
            display = cv2.resize(frame, (save_w, save_h))
            h, w = display.shape[:2]

            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det['pixel']
                # Scale to display size
                x1 = int(x1 * save_w / frame.shape[1])
                y1 = int(y1 * save_h / frame.shape[0])
                x2 = int(x2 * save_w / frame.shape[1])
                y2 = int(y2 * save_h / frame.shape[0])

                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"{det['score']:.2f}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Progress bar
            progress = min(captured / target, 1.0)
            bar_w = 300
            cv2.rectangle(display, (20, 20), (20 + bar_w, 50), (50, 50, 50), -1)
            cv2.rectangle(display, (20, 20), (20 + int(bar_w * progress), 50), (0, 255, 0), -1)
            cv2.putText(display, f"{captured}/{target}", (330, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Mode & threshold
            mode_text = f"AUTO ({auto_interval:.1f}s)" if auto_mode else "MANUAL"
            mode_color = (0, 255, 255) if auto_mode else (200, 200, 200)
            cv2.putText(display, mode_text, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
            cv2.putText(display, f"Thresh: {confidence:.2f}", (200, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Detection count
            cv2.putText(display, f"Boxes: {len(detections)}", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Capture flash
            if time.time() - last_capture < 0.3:
                cv2.rectangle(display, (0, 0), (w, h), (0, 255, 0), 10)

            # Help
            cv2.rectangle(display, (0, h-35), (w, h), (30, 30, 30), -1)
            cv2.putText(display, "SPACE:Capture | A:Auto | +/-:Thresh | Q:Quit",
                       (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Auto-Labeler", display)

            # Keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                save_sample(frame, detections)
            elif key == ord('a'):
                auto_mode = not auto_mode
                last_auto = time.time()
                print(f"  Auto-mode: {'ON' if auto_mode else 'OFF'}")
            elif key in (ord('+'), ord('=')):
                confidence = min(0.5, confidence + 0.05)
                print(f"  Threshold: {confidence:.2f}")
            elif key in (ord('-'), ord('_')):
                confidence = max(0.05, confidence - 0.05)
                print(f"  Threshold: {confidence:.2f}")

            if captured >= target:
                print(f"\n  Target reached!")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Summary
    print("\n" + "=" * 50)
    print("  AUTO-LABELING COMPLETE")
    print("=" * 50)
    print(f"  Images: {captured}")
    print(f"  Labels: {captured} (YOLO format)")
    print(f"  Location: {os.path.abspath(CONFIG['output_dir'])}/")
    print("\n  NEXT: Train YOLOv8")
    print("    python train_yolov8.py")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_label_with_gdino.py <camera_ip>")
    else:
        run(sys.argv[1])
