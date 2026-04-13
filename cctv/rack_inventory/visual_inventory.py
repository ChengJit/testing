#!/usr/bin/env python3
"""
Visual Rack Inventory with Detection Boxes and On-Screen Logging
"""

import cv2
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import multiprocessing as mp
import ctypes
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

from .config import RackInventoryConfig, CameraSource
from .detector import RackInventoryDetector, Detection, ShelfZone


def camera_worker(
    name: str,
    url: str,
    frame_queue: mp.Queue,
    running: mp.Value,
    connected: mp.Value,
):
    """Camera capture worker process."""
    cap = None
    reconnect_delay = 2.0

    while running.value:
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                connected.value = 0
                time.sleep(reconnect_delay)
                continue

            connected.value = 1

            while running.value and cap.isOpened():
                cap.grab()
                ret, frame = cap.retrieve()

                if not ret:
                    connected.value = 0
                    break

                # Clear and put latest
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except:
                        break

                try:
                    # Resize for efficiency
                    h, w = frame.shape[:2]
                    if w > 640:
                        scale = 640 / w
                        frame = cv2.resize(frame, (640, int(h * scale)))
                    frame_queue.put_nowait((frame, time.time()))
                except:
                    pass

                time.sleep(0.03)

        except:
            connected.value = 0
        finally:
            if cap:
                cap.release()

        if running.value:
            time.sleep(reconnect_delay)


class VisualInventoryApp:
    """
    Visual inventory app with detection boxes and on-screen logging.
    """

    def __init__(
        self,
        config: RackInventoryConfig,
        classes: List[str] = None,
        detect_interval: float = 2.0,
    ):
        self.config = config
        self.detect_interval = detect_interval

        if classes is None:
            classes = ["cardboard box", "box", "package", "terminal", "device"]
        self.classes = classes

        # Camera processes
        self.processes: Dict[str, mp.Process] = {}
        self.queues: Dict[str, mp.Queue] = {}
        self.running_flags: Dict[str, mp.Value] = {}
        self.connected_flags: Dict[str, mp.Value] = {}

        # Detector
        self.detector: Optional[RackInventoryDetector] = None

        # State
        self._running = False
        self._last_frames: Dict[str, np.ndarray] = {}
        self._last_detections: Dict[str, List[Detection]] = {}
        self._last_counts: Dict[str, Dict[str, int]] = {}
        self._last_detect_time: Dict[str, float] = {}
        self._prev_counts: Dict[str, Dict[str, int]] = {}

        # On-screen log (last N messages)
        self._log_messages: deque = deque(maxlen=15)

        # Logging to file
        self.log_dir = Path("rack_data")
        self.log_dir.mkdir(exist_ok=True)
        self.json_log = self.log_dir / f"inventory_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    def log(self, message: str, level: str = "INFO"):
        """Add log message to on-screen display and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self._log_messages.append((level, log_entry))
        print(f"{timestamp} [{level}] {message}")

    def log_detection_to_file(self, camera: str, detections: List[Detection], counts: Dict[str, int]):
        """Save detection to JSON log file."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "camera": camera,
            "total": len(detections),
            "counts": {k: int(v) for k, v in counts.items()},
            "detections": [
                {"label": d.label, "conf": round(float(d.confidence), 2), "bbox": [int(x) for x in d.bbox]}
                for d in detections
            ],
        }
        with open(self.json_log, "a") as f:
            f.write(json.dumps(record) + "\n")

    def start(self):
        """Start all components."""
        self.log("Starting Rack Inventory System...")

        # Start cameras
        for cam in self.config.cameras:
            if not cam.enabled:
                continue

            frame_queue = mp.Queue(maxsize=2)
            running_flag = mp.Value(ctypes.c_bool, True)
            connected_flag = mp.Value(ctypes.c_bool, False)

            proc = mp.Process(
                target=camera_worker,
                args=(cam.name, cam.rtsp_url, frame_queue, running_flag, connected_flag),
                daemon=True,
            )
            proc.start()

            self.processes[cam.name] = proc
            self.queues[cam.name] = frame_queue
            self.running_flags[cam.name] = running_flag
            self.connected_flags[cam.name] = connected_flag
            self._last_detect_time[cam.name] = 0

            self.log(f"Camera started: {cam.name} ({cam.ip})")

        # Load detector
        self.log("Loading YOLO-World detector...")
        self.detector = RackInventoryDetector(use_yolo_world=True, model_size="s")
        if self.detector.load(self.classes):
            self.log(f"Detector ready: {', '.join(self.classes)}")
        else:
            self.log("Detector failed to load!", "ERROR")

        self._running = True
        self.log(f"Logging to: {self.json_log}")

    def stop(self):
        """Stop all components."""
        self._running = False
        for flag in self.running_flags.values():
            flag.value = False
        for proc in self.processes.values():
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
        self.log("System stopped")

    def draw_detections(self, frame: np.ndarray, detections: List[Detection], cam_name: str) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        display = frame.copy()

        # Color map for different classes
        colors = {
            "cardboard box": (0, 255, 0),    # Green
            "box": (0, 255, 0),               # Green
            "package": (0, 200, 255),         # Orange
            "terminal": (255, 0, 0),          # Blue
            "device": (255, 0, 255),          # Magenta
        }
        default_color = (0, 255, 255)  # Yellow

        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            color = colors.get(det.label, default_color)

            # Draw box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{det.label} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(display, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Draw counts in corner
        counts = self._last_counts.get(cam_name, {})
        if counts:
            y = 50
            cv2.rectangle(display, (5, 35), (120, 35 + len(counts) * 18 + 10), (0, 0, 0), -1)
            for zone, count in counts.items():
                cv2.putText(display, f"{zone}: {count}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y += 18

        return display

    def draw_log_panel(self, grid: np.ndarray) -> np.ndarray:
        """Draw on-screen log panel on the right side."""
        h, w = grid.shape[:2]
        panel_w = 350

        # Create panel
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # Dark gray

        # Title
        cv2.putText(panel, "INVENTORY LOG", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.line(panel, (10, 35), (panel_w - 10, 35), (100, 100, 100), 1)

        # Log messages
        y = 55
        for level, msg in self._log_messages:
            # Color by level
            if level == "ERROR":
                color = (0, 0, 255)
            elif level == "WARN":
                color = (0, 165, 255)
            elif "CHANGE" in msg or "+" in msg or "-" in msg:
                color = (0, 255, 255)  # Yellow for changes
            else:
                color = (200, 200, 200)

            # Wrap long messages
            if len(msg) > 40:
                msg = msg[:40] + "..."

            cv2.putText(panel, msg, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y += 18

            if y > h - 50:
                break

        # Stats at bottom
        cv2.line(panel, (10, h - 45), (panel_w - 10, h - 45), (100, 100, 100), 1)

        total_items = sum(
            counts.get("Total", 0)
            for counts in self._last_counts.values()
        )
        connected = sum(1 for f in self.connected_flags.values() if f.value)

        cv2.putText(panel, f"Cameras: {connected}/{len(self.config.cameras)}", (10, h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.putText(panel, f"Total Items: {total_items}", (180, h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Combine grid + panel
        combined = np.hstack([grid, panel])
        return combined

    def run_detection(self, cam_name: str, frame: np.ndarray):
        """Run detection and check for changes."""
        if not self.detector or not self.detector.loaded:
            return

        # Auto-detect shelves if needed
        if not self.detector.shelf_zones:
            zones = self.detector.auto_detect_shelves(frame)
            if zones:
                self.detector.set_shelf_zones(zones)
                self.log(f"[{cam_name}] Found {len(zones)} shelf zones")

        # Detect
        detections, counts = self.detector.detect(frame)
        self._last_detections[cam_name] = detections
        self._last_detect_time[cam_name] = time.time()

        # Log to file
        self.log_detection_to_file(cam_name, detections, counts)

        # Check for changes
        prev = self._prev_counts.get(cam_name, {})
        changes = []
        for zone, count in counts.items():
            old = prev.get(zone, count)  # Use current as baseline first time
            diff = count - old
            if diff != 0:
                sign = "+" if diff > 0 else ""
                changes.append(f"{zone}:{sign}{diff}")

        if changes:
            self.log(f"[{cam_name}] CHANGE: {', '.join(changes)}", "WARN")
        else:
            self.log(f"[{cam_name}] Detected {len(detections)} items")

        self._prev_counts[cam_name] = counts.copy()
        self._last_counts[cam_name] = counts

    def create_display(self) -> np.ndarray:
        """Create display with detection boxes."""
        frames = []

        for cam in self.config.cameras:
            name = cam.name

            # Get frame
            try:
                data = self.queues[name].get_nowait()
                frame, ts = data
                self._last_frames[name] = frame.copy()
            except:
                frame = self._last_frames.get(name)

            if frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"{name}: Connecting...", (20, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            else:
                # Check if time to detect
                elapsed = time.time() - self._last_detect_time.get(name, 0)
                if elapsed >= self.detect_interval:
                    self.run_detection(name, frame)

                # Draw detections
                detections = self._last_detections.get(name, [])
                frame = self.draw_detections(frame, detections, name)

                # Add header
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
                connected = self.connected_flags.get(name, mp.Value(ctypes.c_bool, False)).value
                color = (0, 255, 0) if connected else (0, 0, 255)
                cv2.circle(frame, (15, 18), 6, color, -1)
                cv2.putText(frame, name, (30, 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Detection count
                det_count = len(self._last_detections.get(name, []))
                cv2.putText(frame, f"{det_count} items", (frame.shape[1] - 80, 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            frames.append(frame)

        if not frames:
            return np.zeros((360, 640, 3), dtype=np.uint8)

        # Build grid
        cols = min(2, len(frames))
        rows = (len(frames) + cols - 1) // cols

        target_h, target_w = 360, 640
        resized = [cv2.resize(f, (target_w, target_h)) if f.shape[:2] != (target_h, target_w) else f for f in frames]

        while len(resized) < rows * cols:
            resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

        grid_rows = []
        for r in range(rows):
            row = np.hstack(resized[r * cols:(r + 1) * cols])
            grid_rows.append(row)

        grid = np.vstack(grid_rows)

        # Add log panel
        display = self.draw_log_panel(grid)

        # Bottom bar
        h = display.shape[0]
        cv2.rectangle(display, (0, h - 25), (display.shape[1], h), (40, 40, 40), -1)
        cv2.putText(display, "Q:Quit | D:Detect Now | R:Reset Zones | S:Screenshot",
                   (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        return display

    def run(self):
        """Main loop."""
        self.start()

        window_name = "Rack Inventory - Visual Mode"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while self._running:
                display = self.create_display()
                cv2.imshow(window_name, display)

                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('d'):
                    # Force detection now
                    for cam in self.config.cameras:
                        self._last_detect_time[cam.name] = 0
                    self.log("Manual detection triggered")
                elif key == ord('r'):
                    if self.detector:
                        self.detector.shelf_zones = []
                    self.log("Shelf zones reset")
                elif key == ord('s'):
                    fname = f"inventory_{int(time.time())}.jpg"
                    cv2.imwrite(fname, display)
                    self.log(f"Screenshot: {fname}")

        finally:
            self.stop()
            cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="rack_config.json")
    parser.add_argument("--camera", "-cam", action="append")
    parser.add_argument("--classes", help="Comma-separated classes")
    parser.add_argument("--interval", "-i", type=float, default=2.0)

    args = parser.parse_args()

    config = RackInventoryConfig.load(args.config)

    if args.camera:
        for i, ip in enumerate(args.camera):
            if ip.startswith('.'):
                ip = f"192.168.122{ip}"
            config.cameras.append(CameraSource(name=f"Rack-Cam-{i+1}", ip=ip))

    if not config.cameras:
        print("No cameras! Use: --camera .128")
        return

    classes = None
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",")]

    app = VisualInventoryApp(config=config, classes=classes, detect_interval=args.interval)
    app.run()


if __name__ == "__main__":
    main()
