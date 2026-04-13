#!/usr/bin/env python3
"""
Rack Inventory Application with Zero-Shot Detection
====================================================

Real-time inventory tracking using YOLO-World or GroundingDINO.

Usage:
    python -m rack_inventory.inventory_app
    python -m rack_inventory.inventory_app --classes "box,terminal,device"
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import csv

from .config import RackInventoryConfig, CameraSource
from .realtime_viewer import camera_worker
from .detector import RackInventoryDetector, Detection, ShelfZone
import multiprocessing as mp
import ctypes


class InventoryTracker:
    """
    Tracks inventory counts over time and detects changes.
    """

    def __init__(self, log_dir: str = "rack_data"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.current_counts: Dict[str, Dict[str, int]] = {}  # camera -> zone -> count
        self.last_counts: Dict[str, Dict[str, int]] = {}
        self.changes: List[Dict] = []

        # CSV log
        self.csv_path = self.log_dir / f"inventory_{datetime.now():%Y%m%d}.csv"
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV log file."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "camera", "zone", "count", "change"])

    def update(self, camera: str, counts: Dict[str, int]):
        """Update counts for a camera."""
        self.last_counts[camera] = self.current_counts.get(camera, {}).copy()
        self.current_counts[camera] = counts.copy()

        # Detect changes
        for zone, count in counts.items():
            old_count = self.last_counts.get(camera, {}).get(zone, 0)
            change = count - old_count

            if change != 0:
                self.changes.append({
                    "time": datetime.now(),
                    "camera": camera,
                    "zone": zone,
                    "old": old_count,
                    "new": count,
                    "change": change,
                })

                # Log to CSV
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        camera,
                        zone,
                        count,
                        change,
                    ])

    def get_recent_changes(self, limit: int = 5) -> List[Dict]:
        """Get recent inventory changes."""
        return self.changes[-limit:]


class RackInventoryApp:
    """
    Main rack inventory application with detection.
    """

    def __init__(
        self,
        config: RackInventoryConfig,
        classes: List[str] = None,
        detect_interval: float = 2.0,  # seconds between detections
    ):
        self.config = config
        self.detect_interval = detect_interval

        # Default classes for warehouse
        if classes is None:
            classes = [
                "cardboard box",
                "box",
                "package",
                "terminal device",
                "electronic device",
            ]

        self.classes = classes

        # Camera processes
        self.processes: Dict[str, mp.Process] = {}
        self.queues: Dict[str, mp.Queue] = {}
        self.running_flags: Dict[str, mp.Value] = {}
        self.connected_flags: Dict[str, mp.Value] = {}

        # Detector (one per camera or shared)
        self.detectors: Dict[str, RackInventoryDetector] = {}

        # Tracking
        self.tracker = InventoryTracker()

        # State
        self._running = False
        self._last_frames: Dict[str, np.ndarray] = {}
        self._last_detections: Dict[str, List[Detection]] = {}
        self._last_counts: Dict[str, Dict[str, int]] = {}
        self._last_detect_time: Dict[str, float] = {}

    def start(self):
        """Start all components."""
        print("=" * 50)
        print("  Rack Inventory System")
        print("=" * 50)
        print()

        # Start camera processes
        print("Starting cameras...")
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

            print(f"  [{cam.name}] Started ({cam.ip})")

        # Load detector
        print()
        print("Loading detector...")
        detector = RackInventoryDetector(use_yolo_world=True, model_size="s")
        if detector.load(self.classes):
            # Share detector for all cameras
            for cam in self.config.cameras:
                self.detectors[cam.name] = detector
            print(f"  Classes: {self.classes}")
        else:
            print("  WARNING: Detector failed to load!")

        self._running = True
        print()
        print("Ready! Press 'D' to toggle detection, 'Q' to quit")
        print("-" * 50)

    def stop(self):
        """Stop all components."""
        self._running = False

        for flag in self.running_flags.values():
            flag.value = False

        for proc in self.processes.values():
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()

        print("Stopped.")

    def run_detection(self, cam_name: str, frame: np.ndarray):
        """Run detection on a frame."""
        detector = self.detectors.get(cam_name)
        if detector is None or not detector.loaded:
            return

        # Auto-detect shelf zones if not set
        if not detector.shelf_zones:
            zones = detector.auto_detect_shelves(frame)
            if zones:
                detector.set_shelf_zones(zones)
                print(f"  [{cam_name}] Auto-detected {len(zones)} shelf zones")

        # Run detection
        detections, counts = detector.detect(frame)
        self._last_detections[cam_name] = detections
        self._last_counts[cam_name] = counts
        self._last_detect_time[cam_name] = time.time()

        # Update tracker
        self.tracker.update(cam_name, counts)

    def create_display(self) -> np.ndarray:
        """Create display grid with detections."""
        frames = []

        for cam in self.config.cameras:
            name = cam.name

            # Get latest frame
            try:
                data = self.queues[name].get_nowait()
                frame, fps, ts = data
                self._last_frames[name] = frame.copy()
            except:
                frame = self._last_frames.get(name)

            if frame is None:
                # Placeholder
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(frame, name, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                cv2.putText(frame, "Connecting...", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            else:
                # Check if time to detect
                elapsed = time.time() - self._last_detect_time.get(name, 0)
                if elapsed >= self.detect_interval:
                    self.run_detection(name, frame)

                # Draw detections
                detector = self.detectors.get(name)
                detections = self._last_detections.get(name, [])
                counts = self._last_counts.get(name, {})

                if detector and detections:
                    frame = detector.draw_detections(frame, detections, counts)

                # Add header
                frame = frame.copy()
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
                connected = self.connected_flags.get(name, mp.Value(ctypes.c_bool, False)).value
                color = (0, 255, 0) if connected else (0, 0, 255)
                cv2.circle(frame, (15, 18), 6, color, -1)
                cv2.putText(frame, name, (30, 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            frames.append(frame)

        if not frames:
            return np.zeros((360, 640, 3), dtype=np.uint8)

        # Build grid
        cols = self.config.grid_columns
        rows = (len(frames) + cols - 1) // cols

        # Resize all to same size
        target_h, target_w = 360, 640
        resized = [cv2.resize(f, (target_w, target_h)) for f in frames]

        while len(resized) < rows * cols:
            resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

        grid_rows = []
        for r in range(rows):
            row = np.hstack(resized[r*cols:(r+1)*cols])
            grid_rows.append(row)

        grid = np.vstack(grid_rows)

        # Add info bar
        h = grid.shape[0]
        cv2.rectangle(grid, (0, h-30), (grid.shape[1], h), (40, 40, 40), -1)

        # Show recent changes
        changes = self.tracker.get_recent_changes(3)
        if changes:
            change_text = " | ".join([
                f"{c['zone']}: {c['change']:+d}" for c in changes[-3:]
            ])
            cv2.putText(grid, f"Changes: {change_text}", (10, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(grid, "D:Detect Q:Quit S:Screenshot", (grid.shape[1]-250, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        return grid

    def run(self):
        """Main application loop."""
        self.start()

        window_name = "Rack Inventory"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        detection_enabled = True

        try:
            while self._running:
                grid = self.create_display()
                cv2.imshow(window_name, grid)

                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('d'):
                    detection_enabled = not detection_enabled
                    self.detect_interval = 2.0 if detection_enabled else 9999
                    print(f"Detection: {'ON' if detection_enabled else 'OFF'}")
                elif key == ord('s'):
                    fname = f"inventory_{int(time.time())}.jpg"
                    cv2.imwrite(fname, grid)
                    print(f"Saved: {fname}")
                elif key == ord('r'):
                    # Force re-detect shelves
                    for det in self.detectors.values():
                        det.shelf_zones = []
                    print("Reset shelf zones - will auto-detect on next frame")

        finally:
            self.stop()
            cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rack Inventory with Detection")
    parser.add_argument("--config", "-c", default="rack_config.json")
    parser.add_argument("--classes", help="Comma-separated classes to detect")
    parser.add_argument("--interval", "-i", type=float, default=2.0,
                       help="Detection interval in seconds")
    parser.add_argument("--camera", "-cam", action="append",
                       help="Camera IP (can use multiple)")

    args = parser.parse_args()

    # Load config
    config = RackInventoryConfig.load(args.config)

    # Add cameras from CLI
    if args.camera:
        for i, ip in enumerate(args.camera):
            if ip.startswith('.'):
                ip = f"192.168.122{ip}"
            config.cameras.append(CameraSource(name=f"Cam-{i+1}", ip=ip))

    if not config.cameras:
        print("No cameras configured!")
        print("Use: python -m rack_inventory.inventory_app --camera 192.168.122.128")
        return

    # Parse classes
    classes = None
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",")]

    # Run app
    app = RackInventoryApp(
        config=config,
        classes=classes,
        detect_interval=args.interval,
    )
    app.run()


if __name__ == "__main__":
    main()
