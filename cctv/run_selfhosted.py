#!/usr/bin/env python3
"""
Self-Hosted Rack Inventory System
==================================

Fully local AI inventory tracking:
- No cloud required
- YOLO-World detection
- SQLite database
- Web dashboard at http://localhost:5000

Usage:
    python run_selfhosted.py .128 .129
    python run_selfhosted.py .128 .129 --interval 5
"""

import sys
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rack_inventory.config import RackInventoryConfig, CameraSource
from rack_inventory.self_hosted import SelfHostedInventory


def main():
    if len(sys.argv) < 2:
        print("Self-Hosted Rack Inventory System")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python run_selfhosted.py <camera_ips>")
        print()
        print("Examples:")
        print("  python run_selfhosted.py .128 .129")
        print("  python run_selfhosted.py .128 --interval 5")
        print()
        print("Features:")
        print("  - Local AI detection (YOLO-World)")
        print("  - SQLite database tracking")
        print("  - Web dashboard at http://localhost:5000")
        print("  - Change detection & alerts")
        print()
        print("Requirements:")
        print("  pip install ultralytics flask")
        return

    config = RackInventoryConfig()
    interval = 3.0

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--interval", "-i") and i + 1 < len(args):
            interval = float(args[i + 1])
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
        print("No cameras specified!")
        return

    system = SelfHostedInventory(
        config=config,
        detect_interval=interval,
    )
    system.run()


if __name__ == "__main__":
    main()
