#!/usr/bin/env python3
"""
Self-Hosted Rack Inventory System
==================================

Fully local AI inventory tracking with:
- YOLO-World detection (no cloud)
- SQLite database
- Web dashboard
- Change detection & alerts

Runs on Windows (dev) or Jetson (production).
"""

import cv2
import numpy as np
import time
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import multiprocessing as mp
import ctypes
import os

# Suppress warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

from .config import RackInventoryConfig, CameraSource
from .detector import RackInventoryDetector, Detection
from .inventory_db import InventoryDatabase


def camera_worker(name, url, frame_queue, running, connected):
    """Camera capture process."""
    cap = None
    while running.value:
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                connected.value = 0
                time.sleep(2)
                continue
            connected.value = 1
            while running.value and cap.isOpened():
                cap.grab()
                ret, frame = cap.retrieve()
                if not ret:
                    connected.value = 0
                    break
                while not frame_queue.empty():
                    try: frame_queue.get_nowait()
                    except: break
                try:
                    h, w = frame.shape[:2]
                    if w > 640:
                        scale = 640 / w
                        frame = cv2.resize(frame, (640, int(h * scale)))
                    frame_queue.put_nowait((frame, time.time()))
                except: pass
                time.sleep(0.05)
        except:
            connected.value = 0
        finally:
            if cap: cap.release()
        if running.value:
            time.sleep(2)


class SelfHostedInventory:
    """
    Self-hosted inventory system with detection and database.
    """

    def __init__(
        self,
        config: RackInventoryConfig,
        detect_interval: float = 3.0,
        db_path: str = "rack_data/inventory.db",
    ):
        self.config = config
        self.detect_interval = detect_interval

        # Database
        self.db = InventoryDatabase(db_path)

        # Detection
        self.detector = None
        self.classes = ["cardboard box", "box", "package", "terminal", "device"]

        # Camera processes
        self.processes: Dict[str, mp.Process] = {}
        self.queues: Dict[str, mp.Queue] = {}
        self.running_flags: Dict[str, mp.Value] = {}
        self.connected_flags: Dict[str, mp.Value] = {}

        # State
        self._running = False
        self._last_frames: Dict[str, np.ndarray] = {}
        self._last_detections: Dict[str, List[Detection]] = {}
        self._last_counts: Dict[str, Dict[str, int]] = {}
        self._last_detect_time: Dict[str, float] = {}

        # Web server
        self._web_thread = None

    def start(self):
        """Start all components."""
        print("=" * 60)
        print("  SELF-HOSTED RACK INVENTORY SYSTEM")
        print("=" * 60)
        print(f"  Database: {self.db.db_path}")
        print(f"  Cameras: {len(self.config.cameras)}")
        print(f"  Detect interval: {self.detect_interval}s")
        print("=" * 60)

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

            print(f"  [+] Camera: {cam.name} ({cam.ip})")

        # Load detector
        print("\n  Loading YOLO-World detector...")
        self.detector = RackInventoryDetector(use_yolo_world=True, model_size="s")
        if self.detector.load(self.classes):
            print(f"  [+] Detector ready")
        else:
            print(f"  [!] Detector failed")

        # Start web dashboard
        self._start_web_server()

        self._running = True
        print("\n" + "=" * 60)
        print("  System running!")
        if self._web_thread:
            print("  Dashboard: http://localhost:5000")
        print("  Logs: Console output + rack_data/inventory.db")
        print("  Press Ctrl+C to stop")
        print("=" * 60 + "\n")

    def stop(self):
        """Stop all components."""
        self._running = False
        for flag in self.running_flags.values():
            flag.value = False
        for proc in self.processes.values():
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
        print("\nSystem stopped.")

    def _start_web_server(self):
        """Start Flask web dashboard in background thread (optional)."""
        try:
            from flask import Flask, jsonify, render_template_string
        except ImportError:
            print("  [!] Flask not installed - running without web dashboard")
            print("      (Install later with: pip install flask --user)")
            return

        try:

            app = Flask(__name__)
            app.config['JSON_SORT_KEYS'] = False

            # Store reference for routes
            inventory_system = self

            DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Rack Inventory</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #00d4ff; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #16213e; padding: 20px; border-radius: 10px; }
        .card h3 { margin-top: 0; color: #00d4ff; }
        .stat { font-size: 2em; font-weight: bold; color: #00ff88; }
        .change { padding: 5px 10px; margin: 5px 0; background: #0f3460; border-radius: 5px; }
        .change.positive { border-left: 3px solid #00ff88; }
        .change.negative { border-left: 3px solid #ff4444; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
        .alert { background: #ff4444; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }
    </style>
</head>
<body>
    <h1>Rack Inventory Dashboard</h1>
    <p>Last updated: {{ now }}</p>

    <div class="grid">
        <div class="card">
            <h3>Summary</h3>
            <p>Total Items: <span class="stat">{{ summary.total_items }}</span></p>
            <p>Changes Today: {{ summary.changes_today }}</p>
            <p>Pending Alerts: {{ summary.pending_alerts }}</p>
        </div>

        <div class="card">
            <h3>By Camera</h3>
            {% for cam, count in summary.by_camera.items() %}
            <p>{{ cam }}: <strong>{{ count }}</strong> items</p>
            {% endfor %}
        </div>

        <div class="card">
            <h3>Recent Changes</h3>
            {% for change in changes %}
            <div class="change {{ 'positive' if change.change > 0 else 'negative' }}">
                {{ change.camera }} / {{ change.shelf }}:
                {{ change.old_count }} → {{ change.new_count }}
                ({{ '+' if change.change > 0 else '' }}{{ change.change }})
                <br><small>{{ change.timestamp }}</small>
            </div>
            {% endfor %}
        </div>

        <div class="card">
            <h3>Current Inventory</h3>
            <table>
                <tr><th>Camera</th><th>Shelf</th><th>Count</th><th>Updated</th></tr>
                {% for item in inventory %}
                <tr>
                    <td>{{ item.camera }}</td>
                    <td>{{ item.shelf }}</td>
                    <td><strong>{{ item.count }}</strong></td>
                    <td>{{ item.timestamp.strftime('%H:%M:%S') }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
    </div>

    {% if alerts %}
    <h2>Alerts</h2>
    {% for alert in alerts %}
    <div class="alert">
        {{ alert.message }} ({{ alert.timestamp }})
    </div>
    {% endfor %}
    {% endif %}
</body>
</html>
'''

            @app.route('/')
            def dashboard():
                return render_template_string(
                    DASHBOARD_HTML,
                    now=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    summary=inventory_system.db.get_summary(),
                    changes=inventory_system.db.get_recent_changes(10),
                    inventory=inventory_system.db.get_current_inventory(),
                    alerts=inventory_system.db.get_unacknowledged_alerts(),
                )

            @app.route('/api/summary')
            def api_summary():
                return jsonify(inventory_system.db.get_summary())

            @app.route('/api/inventory')
            def api_inventory():
                inv = inventory_system.db.get_current_inventory()
                return jsonify([{
                    'camera': i.camera,
                    'shelf': i.shelf,
                    'count': i.count,
                    'timestamp': i.timestamp.isoformat()
                } for i in inv])

            @app.route('/api/changes')
            def api_changes():
                changes = inventory_system.db.get_recent_changes(50)
                return jsonify([{
                    'camera': c.camera,
                    'shelf': c.shelf,
                    'old': c.old_count,
                    'new': c.new_count,
                    'change': c.change,
                    'timestamp': c.timestamp.isoformat()
                } for c in changes])

            def run_flask():
                import logging
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.ERROR)
                app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

            self._web_thread = threading.Thread(target=run_flask, daemon=True)
            self._web_thread.start()
            print("  [+] Web dashboard started on http://localhost:5000")

        except Exception as e:
            print(f"  [!] Web dashboard error: {e}")

    def run_detection(self, cam_name: str, frame: np.ndarray):
        """Run detection and update database."""
        if not self.detector or not self.detector.loaded:
            return

        # Auto-detect shelves
        if not self.detector.shelf_zones:
            zones = self.detector.auto_detect_shelves(frame)
            if zones:
                self.detector.set_shelf_zones(zones)
                print(f"[{cam_name}] Found {len(zones)} shelf zones")

        # Detect
        detections, counts = self.detector.detect(frame)
        self._last_detections[cam_name] = detections
        self._last_counts[cam_name] = counts
        self._last_detect_time[cam_name] = time.time()

        # Update database
        for shelf, count in counts.items():
            if shelf == "Total":
                continue
            change = self.db.update_count(cam_name, shelf, int(count))
            if change:
                sign = "+" if change > 0 else ""
                print(f"[{cam_name}] {shelf}: {sign}{change} (now: {count})")

    def run(self):
        """Main loop."""
        self.start()

        try:
            while self._running:
                # Process each camera
                for cam in self.config.cameras:
                    name = cam.name

                    # Check if time to detect
                    if time.time() - self._last_detect_time.get(name, 0) < self.detect_interval:
                        continue

                    # Get frame
                    try:
                        frame, ts = self.queues[name].get(timeout=0.1)
                        self._last_frames[name] = frame
                    except:
                        frame = self._last_frames.get(name)

                    if frame is None:
                        continue

                    # Skip old frames
                    if time.time() - ts > 2:
                        continue

                    # Run detection
                    self.run_detection(name, frame)

                # Status every 30 seconds
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Self-Hosted Rack Inventory")
    parser.add_argument("--config", "-c", default="rack_config.json")
    parser.add_argument("--camera", "-cam", action="append")
    parser.add_argument("--interval", "-i", type=float, default=3.0)
    parser.add_argument("--db", default="rack_data/inventory.db")

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

    system = SelfHostedInventory(
        config=config,
        detect_interval=args.interval,
        db_path=args.db,
    )
    system.run()


if __name__ == "__main__":
    main()
