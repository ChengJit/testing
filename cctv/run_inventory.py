#!/usr/bin/env python3
"""
Quick launcher for Rack Inventory with Detection.

Usage:
    python run_inventory.py                           # Use saved config
    python run_inventory.py .128 .129                 # Camera IPs (shorthand)
    python run_inventory.py --classes "box,terminal"  # Custom classes
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rack_inventory.config import RackInventoryConfig, CameraSource
from rack_inventory.inventory_app import RackInventoryApp


def main():
    config = RackInventoryConfig()
    classes = None
    interval = 2.0

    # Parse args
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "--classes" and i + 1 < len(args):
            classes = [c.strip() for c in args[i+1].split(",")]
            i += 2
            continue
        elif arg == "--interval" and i + 1 < len(args):
            interval = float(args[i+1])
            i += 2
            continue
        elif arg.startswith("-"):
            i += 1
            continue

        # Camera IP
        ip = arg
        if ip.startswith('.'):
            ip = f"192.168.122{ip}"

        config.cameras.append(CameraSource(
            name=f"Rack-Cam-{len(config.cameras)+1}",
            ip=ip,
        ))
        i += 1

    # Load from config if no cameras specified
    if not config.cameras:
        try:
            config = RackInventoryConfig.load("rack_config.json")
            print(f"Loaded config: {len(config.cameras)} cameras")
        except:
            pass

    if not config.cameras:
        print("Usage: python run_inventory.py <camera_ips>")
        print()
        print("Examples:")
        print("  python run_inventory.py 192.168.122.128 192.168.122.129")
        print("  python run_inventory.py .128 .129  (shorthand)")
        print("  python run_inventory.py --classes 'box,terminal,device'")
        return

    print(f"Starting with {len(config.cameras)} camera(s):")
    for cam in config.cameras:
        print(f"  - {cam.name}: {cam.ip}")
    print()

    app = RackInventoryApp(
        config=config,
        classes=classes,
        detect_interval=interval,
    )
    app.run()


if __name__ == "__main__":
    main()
