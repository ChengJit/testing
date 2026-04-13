#!/usr/bin/env python3
"""
Training Data Collector Launcher

Usage:
    python run_training.py .128              # Collect from camera
    python run_training.py .128 --export     # Export to YOLO format
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rack_inventory.training_collector import TrainingCollector


def main():
    if len(sys.argv) < 2:
        print("Training Data Collector for Rack Inventory")
        print("=" * 45)
        print()
        print("Usage:")
        print("  python run_training.py <camera_ip>")
        print("  python run_training.py <camera_ip> --export")
        print()
        print("Examples:")
        print("  python run_training.py .128          # Start collecting")
        print("  python run_training.py .128 --export # Export YOLO dataset")
        print()
        print("Workflow:")
        print("  1. Start with EMPTY rack")
        print("  2. Press B to capture BASELINE")
        print("  3. Add items to rack")
        print("  4. Press C to capture frame")
        print("  5. Draw boxes around items (left-click drag)")
        print("  6. Press S to save")
        print("  7. Repeat steps 3-6")
        print("  8. Export with --export when done")
        return

    camera = sys.argv[1]
    export_only = "--export" in sys.argv

    # Build RTSP URL
    if not camera.startswith("rtsp://"):
        if camera.startswith("."):
            camera = f"192.168.122{camera}"
        camera_url = f"rtsp://fasspay:fasspay2025@{camera}:554/stream2?rtsp_transport=tcp"
    else:
        camera_url = camera

    collector = TrainingCollector(output_dir="rack_training_data")

    if export_only:
        collector.export_yolo_dataset()
    else:
        print(f"Connecting to: {camera_url}")
        collector.capture_from_camera(camera_url)


if __name__ == "__main__":
    main()
