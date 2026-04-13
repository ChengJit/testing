#!/usr/bin/env python3
"""
Training Image Collector for YOLOv8
Captures images from your CCTV for training a custom box detector.

Usage:
    python collect_training_images.py <camera_ip>

Controls:
    SPACE  - Capture current frame
    A      - Auto-capture mode (captures every few seconds)
    +/-    - Adjust auto-capture interval
    Q      - Quit and show summary
"""

import cv2
import os
import sys
import time
import threading
from datetime import datetime

# ============ CONFIGURATION ============
CONFIG = {
    "rtsp_user": "fasspay",
    "rtsp_pass": "fasspay2025",
    "use_hd": True,

    "output_dir": "dataset/images",
    "auto_interval": 3.0,        # Seconds between auto-captures
    "target_images": 100,        # Goal number of images

    "resize_width": 1280,        # Resize for training (None = keep original)
    "resize_height": 720,
}
# =======================================


class FrameGrabber:
    def __init__(self, url):
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        # Try GStreamer first
        gst = (
            f"rtspsrc location={url} latency=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )
        self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print("  GStreamer failed, using FFmpeg...")
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

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


def run(camera_ip):
    print("=" * 50)
    print("  TRAINING IMAGE COLLECTOR")
    print("=" * 50)

    # Create output directory
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Count existing images
    existing = len([f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png'))])
    print(f"\nOutput directory: {output_dir}")
    print(f"Existing images: {existing}")
    print(f"Target: {CONFIG['target_images']} images")

    # Connect to camera
    stream = "stream1" if CONFIG["use_hd"] else "stream2"
    url = f"rtsp://{CONFIG['rtsp_user']}:{CONFIG['rtsp_pass']}@{camera_ip}:554/{stream}"

    print(f"\nConnecting to camera...")
    cap = FrameGrabber(url)
    time.sleep(1)

    ret, frame = cap.read()
    if not ret:
        print("Failed to connect!")
        return

    print(f"Connected! Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print("\n" + "=" * 50)
    print("  CONTROLS")
    print("=" * 50)
    print("  SPACE  - Capture frame")
    print("  A      - Toggle auto-capture")
    print("  +/-    - Adjust auto interval")
    print("  Q      - Quit")
    print("=" * 50)
    print("\nTIPS for good training data:")
    print("  - Vary box positions and arrangements")
    print("  - Include different lighting conditions")
    print("  - Capture boxes at different distances")
    print("  - Include some edge cases (partial boxes, stacked)")
    print("=" * 50 + "\n")

    cv2.namedWindow("Collector", cv2.WINDOW_NORMAL)

    captured = existing
    auto_mode = False
    auto_interval = CONFIG["auto_interval"]
    last_auto = 0
    last_capture_time = 0

    target = CONFIG["target_images"]
    resize_w = CONFIG.get("resize_width")
    resize_h = CONFIG.get("resize_height")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Resize if configured
            if resize_w and resize_h:
                display = cv2.resize(frame, (resize_w, resize_h))
            else:
                display = frame.copy()

            # Draw UI
            h, w = display.shape[:2]

            # Progress bar
            progress = min(captured / target, 1.0)
            bar_w = 300
            cv2.rectangle(display, (20, 20), (20 + bar_w, 50), (50, 50, 50), -1)
            cv2.rectangle(display, (20, 20), (20 + int(bar_w * progress), 50), (0, 255, 0), -1)
            cv2.putText(display, f"{captured}/{target}", (330, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Mode indicator
            mode_text = f"AUTO ({auto_interval:.1f}s)" if auto_mode else "MANUAL"
            mode_color = (0, 255, 255) if auto_mode else (200, 200, 200)
            cv2.putText(display, mode_text, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

            # Capture flash
            if time.time() - last_capture_time < 0.3:
                cv2.rectangle(display, (0, 0), (w, h), (0, 255, 0), 10)

            # Instructions
            cv2.rectangle(display, (0, h-35), (w, h), (30, 30, 30), -1)
            cv2.putText(display, "SPACE:Capture | A:Auto | +/-:Interval | Q:Quit",
                       (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Collector", display)

            # Auto capture
            if auto_mode and time.time() - last_auto >= auto_interval:
                save_frame = cv2.resize(frame, (resize_w, resize_h)) if resize_w else frame
                filename = f"box_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{captured:04d}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, save_frame)
                captured += 1
                last_auto = time.time()
                last_capture_time = time.time()
                print(f"  [AUTO] Saved: {filename} ({captured}/{target})")

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                save_frame = cv2.resize(frame, (resize_w, resize_h)) if resize_w else frame
                filename = f"box_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{captured:04d}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, save_frame)
                captured += 1
                last_capture_time = time.time()
                print(f"  [MANUAL] Saved: {filename} ({captured}/{target})")

            elif key == ord('a'):
                auto_mode = not auto_mode
                last_auto = time.time()
                print(f"  Auto-capture: {'ON' if auto_mode else 'OFF'}")

            elif key in (ord('+'), ord('=')):
                auto_interval = min(10.0, auto_interval + 0.5)
                print(f"  Auto interval: {auto_interval:.1f}s")

            elif key in (ord('-'), ord('_')):
                auto_interval = max(0.5, auto_interval - 0.5)
                print(f"  Auto interval: {auto_interval:.1f}s")

            # Check if target reached
            if captured >= target:
                print(f"\n  Target reached! {captured} images collected.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Summary
    print("\n" + "=" * 50)
    print("  COLLECTION COMPLETE")
    print("=" * 50)
    print(f"  Images saved: {captured}")
    print(f"  Location: {os.path.abspath(output_dir)}")
    print("\n  NEXT STEPS:")
    print("  1. Label images with LabelImg or Roboflow")
    print("     pip install labelImg")
    print("     labelImg dataset/images dataset/labels")
    print("")
    print("  2. Create dataset.yaml:")
    print("     path: dataset")
    print("     train: images")
    print("     val: images")
    print("     names:")
    print("       0: box")
    print("")
    print("  3. Train YOLOv8:")
    print("     python train_yolov8.py")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python collect_training_images.py <camera_ip>")
        print("Example: python collect_training_images.py 192.168.122.128")
    else:
        run(sys.argv[1])
