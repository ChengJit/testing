#!/usr/bin/env python3
"""
Quick launcher for real-time rack viewer.

Usage:
    python run_realtime.py                          # Use saved config
    python run_realtime.py 192.168.122.128 .129    # Direct IPs
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rack_inventory.config import RackInventoryConfig, CameraSource
from rack_inventory.realtime_viewer import run_realtime


def main():
    config = RackInventoryConfig()

    if len(sys.argv) > 1:
        # Use IPs from command line
        for i, arg in enumerate(sys.argv[1:]):
            # Handle shorthand like ".128" -> "192.168.122.128"
            if arg.startswith('.'):
                ip = f"192.168.122{arg}"
            else:
                ip = arg

            config.cameras.append(CameraSource(
                name=f"Cam-{i+1}",
                ip=ip,
            ))
    else:
        # Try to load config
        try:
            config = RackInventoryConfig.load("rack_config.json")
        except:
            print("No config found. Provide camera IPs:")
            print("  python run_realtime.py 192.168.122.128 192.168.122.129")
            print("  python run_realtime.py .128 .129  (shorthand)")
            return

    if not config.cameras:
        print("No cameras configured!")
        return

    print(f"Cameras: {[c.ip for c in config.cameras]}")
    run_realtime(config)


if __name__ == "__main__":
    main()
