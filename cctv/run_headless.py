#!/usr/bin/env python3
"""
Headless Rack Inventory - No Display, Just Logging

Usage:
    python run_headless.py .128 .129
    python run_headless.py .128 .129 --interval 10
"""

import sys
import os

# Suppress OpenCV/FFmpeg warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rack_inventory.config import RackInventoryConfig, CameraSource
from rack_inventory.headless import HeadlessInventoryScanner


def main():
    config = RackInventoryConfig()
    classes = None
    interval = 5.0

    # Parse args
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ("--classes", "-cls") and i + 1 < len(args):
            classes = [c.strip() for c in args[i+1].split(",")]
            i += 2
        elif arg in ("--interval", "-i") and i + 1 < len(args):
            interval = float(args[i+1])
            i += 2
        elif arg.startswith("-"):
            i += 1
        else:
            # Camera IP
            ip = arg
            if ip.startswith('.'):
                ip = f"192.168.122{ip}"
            config.cameras.append(CameraSource(
                name=f"Rack-Cam-{len(config.cameras)+1}",
                ip=ip,
            ))
            i += 1

    # Load config if no cameras
    if not config.cameras:
        try:
            config = RackInventoryConfig.load("rack_config.json")
        except:
            pass

    if not config.cameras:
        print("Usage: python run_headless.py <camera_ips>")
        print()
        print("Examples:")
        print("  python run_headless.py .128 .129")
        print("  python run_headless.py .128 .129 --interval 10")
        print("  python run_headless.py .128 --classes 'box,terminal'")
        return

    print(f"Cameras: {[c.ip for c in config.cameras]}")
    print(f"Interval: {interval}s")
    print()

    scanner = HeadlessInventoryScanner(
        config=config,
        classes=classes,
        detect_interval=interval,
    )
    scanner.run()


if __name__ == "__main__":
    main()
