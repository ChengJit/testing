#!/usr/bin/env python3
"""
Visual Rack Inventory - With Detection Boxes and On-Screen Logging

Usage:
    python run_visual.py .128 .129
    python run_visual.py .128 .129 --interval 3
"""

import sys
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rack_inventory.config import RackInventoryConfig, CameraSource
from rack_inventory.visual_inventory import VisualInventoryApp


def main():
    config = RackInventoryConfig()
    classes = None
    interval = 2.0

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
            ip = arg
            if ip.startswith('.'):
                ip = f"192.168.122{ip}"
            config.cameras.append(CameraSource(
                name=f"Rack-Cam-{len(config.cameras)+1}",
                ip=ip,
            ))
            i += 1

    if not config.cameras:
        try:
            config = RackInventoryConfig.load("rack_config.json")
        except:
            pass

    if not config.cameras:
        print("Usage: python run_visual.py <camera_ips>")
        print("  python run_visual.py .128 .129")
        return

    print(f"Cameras: {[c.ip for c in config.cameras]}")

    app = VisualInventoryApp(
        config=config,
        classes=classes,
        detect_interval=interval,
    )
    app.run()


if __name__ == "__main__":
    main()
