#!/usr/bin/env python3
"""
Headless Rack Inventory Scanner
================================

Runs without display, logs detections to file.
Uses less CPU/GPU by skipping visualization.

Usage:
    python -m rack_inventory.headless --camera .128 .129
    python -m rack_inventory.headless --interval 5
"""

import cv2
import numpy as np
import time
import logging
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import multiprocessing as mp
import ctypes

from .config import RackInventoryConfig, CameraSource
from .detector import RackInventoryDetector, Detection

# Suppress OpenCV/FFmpeg warnings
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rack_inventory.log"),
    ]
)
logger = logging.getLogger(__name__)


def camera_capture_loop(
    name: str,
    url: str,
    frame_queue: mp.Queue,
    running: mp.Value,
    connected: mp.Value,
):
    """Camera capture process - minimal, no display."""
    # Suppress FFmpeg warnings
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

    cap = None
    reconnect_delay = 3.0

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
                # Grab latest, skip buffer
                cap.grab()
                ret, frame = cap.retrieve()

                if not ret:
                    connected.value = 0
                    break

                # Clear queue, put latest
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except:
                        break

                try:
                    frame_queue.put_nowait((frame, time.time()))
                except:
                    pass

                # Slower capture for headless (save CPU)
                time.sleep(0.1)  # ~10 FPS max

        except Exception as e:
            connected.value = 0
        finally:
            if cap:
                cap.release()

        if running.value:
            time.sleep(reconnect_delay)


class HeadlessInventoryScanner:
    """
    Headless inventory scanner with logging.
    No display, minimal CPU usage.
    """

    def __init__(
        self,
        config: RackInventoryConfig,
        classes: List[str] = None,
        detect_interval: float = 5.0,
        log_dir: str = "rack_data",
    ):
        self.config = config
        self.detect_interval = detect_interval
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Default classes
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
        self._last_counts: Dict[str, Dict[str, int]] = {}

        # JSON log file
        self.json_log = self.log_dir / f"inventory_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    def log_detection(self, camera: str, detections: List[Detection], counts: Dict[str, int]):
        """Log detection results to JSON Lines file."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "camera": camera,
            "total_detections": len(detections),
            "counts": {k: int(v) for k, v in counts.items()},  # Convert numpy int64
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(float(d.confidence), 3),
                    "bbox": [int(x) for x in d.bbox],  # Convert numpy ints
                }
                for d in detections
            ],
        }

        with open(self.json_log, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Check for changes
        old_counts = self._last_counts.get(camera, {})
        changes = []
        for zone, count in counts.items():
            old = old_counts.get(zone, 0)
            if count != old:
                changes.append(f"{zone}: {old}->{count}")

        if changes:
            logger.info(f"[{camera}] CHANGE: {', '.join(changes)}")

        self._last_counts[camera] = counts.copy()

    def start(self):
        """Start scanner."""
        logger.info("=" * 50)
        logger.info("  Headless Rack Inventory Scanner")
        logger.info("=" * 50)

        # Start cameras
        for cam in self.config.cameras:
            if not cam.enabled:
                continue

            frame_queue = mp.Queue(maxsize=2)
            running_flag = mp.Value(ctypes.c_bool, True)
            connected_flag = mp.Value(ctypes.c_bool, False)

            proc = mp.Process(
                target=camera_capture_loop,
                args=(cam.name, cam.rtsp_url, frame_queue, running_flag, connected_flag),
                daemon=True,
            )
            proc.start()

            self.processes[cam.name] = proc
            self.queues[cam.name] = frame_queue
            self.running_flags[cam.name] = running_flag
            self.connected_flags[cam.name] = connected_flag

            logger.info(f"[{cam.name}] Started camera process ({cam.ip})")

        # Load detector
        logger.info("Loading YOLO-World detector...")
        self.detector = RackInventoryDetector(use_yolo_world=True, model_size="s")
        if self.detector.load(self.classes):
            logger.info(f"Detector loaded: {self.classes}")
        else:
            logger.error("Failed to load detector!")
            return False

        self._running = True
        logger.info(f"Logging to: {self.json_log}")
        logger.info(f"Detection interval: {self.detect_interval}s")
        logger.info("Press Ctrl+C to stop")
        logger.info("-" * 50)

        return True

    def stop(self):
        """Stop scanner."""
        self._running = False

        for flag in self.running_flags.values():
            flag.value = False

        for name, proc in self.processes.items():
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()

        logger.info("Scanner stopped")
        logger.info(f"Log saved: {self.json_log}")

    def run(self):
        """Main loop."""
        if not self.start():
            return

        # Handle Ctrl+C
        def signal_handler(sig, frame):
            logger.info("Stopping...")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)

        last_detect_time: Dict[str, float] = {cam.name: 0 for cam in self.config.cameras}
        last_status_time = 0

        try:
            while self._running:
                current_time = time.time()

                # Status update every 30 seconds
                if current_time - last_status_time >= 30:
                    connected = sum(1 for f in self.connected_flags.values() if f.value)
                    total = len(self.connected_flags)
                    logger.info(f"Status: {connected}/{total} cameras connected")
                    last_status_time = current_time

                # Process each camera
                for cam in self.config.cameras:
                    name = cam.name

                    # Check if time to detect
                    if current_time - last_detect_time.get(name, 0) < self.detect_interval:
                        continue

                    # Get frame
                    try:
                        frame, ts = self.queues[name].get(timeout=0.1)
                    except:
                        continue

                    # Skip old frames
                    if current_time - ts > 2.0:
                        continue

                    # Auto-detect shelves
                    if not self.detector.shelf_zones:
                        zones = self.detector.auto_detect_shelves(frame)
                        if zones:
                            self.detector.set_shelf_zones(zones)
                            logger.info(f"[{name}] Detected {len(zones)} shelf zones")

                    # Run detection
                    detections, counts = self.detector.detect(frame)
                    last_detect_time[name] = current_time

                    # Log results
                    self.log_detection(name, detections, counts)

                    if detections:
                        logger.info(
                            f"[{name}] Detected {len(detections)} objects: "
                            f"{counts.get('Total', 0)} total"
                        )

                # Sleep to reduce CPU
                time.sleep(0.5)

        finally:
            self.stop()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Headless Rack Inventory Scanner")
    parser.add_argument("--config", "-c", default="rack_config.json")
    parser.add_argument("--camera", "-cam", action="append", help="Camera IP")
    parser.add_argument("--classes", help="Comma-separated detection classes")
    parser.add_argument("--interval", "-i", type=float, default=5.0,
                       help="Detection interval in seconds (default: 5)")
    parser.add_argument("--log-dir", default="rack_data", help="Log directory")

    args = parser.parse_args()

    # Load config
    config = RackInventoryConfig.load(args.config)

    # Add cameras from CLI
    if args.camera:
        for i, ip in enumerate(args.camera):
            if ip.startswith('.'):
                ip = f"192.168.122{ip}"
            config.cameras.append(CameraSource(name=f"Rack-Cam-{i+1}", ip=ip))

    if not config.cameras:
        print("No cameras! Use: --camera 192.168.122.128")
        return

    # Parse classes
    classes = None
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",")]

    # Run
    scanner = HeadlessInventoryScanner(
        config=config,
        classes=classes,
        detect_interval=args.interval,
        log_dir=args.log_dir,
    )
    scanner.run()


if __name__ == "__main__":
    main()
