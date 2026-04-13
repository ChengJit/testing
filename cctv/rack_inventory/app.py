#!/usr/bin/env python3
"""
Rack Inventory Scanner - Main Application
==========================================

Scan and monitor terminal stock on racks using multiple cameras.

Usage:
    python -m rack_inventory                 # Run with existing config
    python -m rack_inventory --scan          # Scan for cameras first
    python -m rack_inventory --setup         # Interactive setup
    python -m rack_inventory --view          # View cameras only (no detection)
"""

import argparse
import sys
from pathlib import Path

from .config import RackInventoryConfig, CameraSource, create_default_config
from .scanner import scan_network, quick_scan, FoundCamera
from .viewer import MultiCameraViewer, run_viewer
from .realtime_viewer import RealtimeMultiViewer, run_realtime


def scan_cameras() -> list:
    """Scan network for cameras."""
    print("=" * 50)
    print("  Scanning for Tapo Cameras")
    print("=" * 50)
    print()

    # Quick scan first
    print("Quick scan of likely IPs (192.168.122.125-140)...")
    likely_ips = [f"192.168.122.{i}" for i in range(125, 145)]
    cameras = quick_scan(likely_ips)

    if not cameras:
        print("\nNo cameras in quick scan. Full network scan...")

        def progress(current, total, ip, found):
            if current % 30 == 0 or found:
                status = "FOUND!" if found else ""
                print(f"  [{current}/{total}] {ip} {status}")

        cameras = scan_network(progress_callback=progress)

    print()
    if cameras:
        print(f"Found {len(cameras)} camera(s):")
        for i, cam in enumerate(cameras, 1):
            print(f"  {i}. {cam.ip} - {cam.rtsp_url}")
    else:
        print("No cameras found!")

    return cameras


def setup_wizard():
    """Interactive setup wizard."""
    print("=" * 50)
    print("  Rack Inventory - Setup Wizard")
    print("=" * 50)
    print()

    # Scan for cameras
    cameras = scan_cameras()

    if not cameras:
        print("\nCannot continue without cameras.")
        print("Make sure cameras are:")
        print("  - Powered on and connected to network")
        print("  - RTSP enabled in Tapo app")
        print("  - Using credentials: fasspay:fasspay2025")
        return

    # Create config
    config = RackInventoryConfig()

    print("\nConfiguring cameras...")
    for i, cam in enumerate(cameras):
        default_name = f"Rack-Cam-{i+1}"
        name = input(f"  Name for {cam.ip} [{default_name}]: ").strip() or default_name

        config.cameras.append(CameraSource(
            name=name,
            ip=cam.ip,
        ))

    # Save config
    config.save("rack_config.json")
    print()
    print("=" * 50)
    print("  Configuration saved to rack_config.json")
    print("  Run 'python -m rack_inventory' to start")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Rack Inventory Scanner - Monitor terminal stock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--scan", action="store_true",
                       help="Scan network for cameras")
    parser.add_argument("--setup", action="store_true",
                       help="Run interactive setup wizard")
    parser.add_argument("--view", action="store_true",
                       help="View cameras only (no detection)")
    parser.add_argument("--realtime", "-rt", action="store_true",
                       help="Use optimized real-time viewer (multiprocessing)")
    parser.add_argument("--config", "-c", default="rack_config.json",
                       help="Config file path")
    parser.add_argument("--camera", "-cam", action="append",
                       help="Add camera by IP (can use multiple times)")

    args = parser.parse_args()

    # Scan mode
    if args.scan:
        cameras = scan_cameras()
        if cameras:
            save = input("\nSave to config? (y/n): ").strip().lower()
            if save == 'y':
                config = RackInventoryConfig()
                for i, cam in enumerate(cameras):
                    config.cameras.append(CameraSource(
                        name=f"Rack-Cam-{i+1}",
                        ip=cam.ip,
                    ))
                config.save(args.config)
                print(f"Saved to {args.config}")
        return

    # Setup mode
    if args.setup:
        setup_wizard()
        return

    # Load or create config
    config_path = Path(args.config)
    if config_path.exists():
        config = RackInventoryConfig.load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = RackInventoryConfig()

    # Add cameras from command line
    if args.camera:
        for i, ip in enumerate(args.camera):
            config.cameras.append(CameraSource(
                name=f"Camera-{i+1}",
                ip=ip,
            ))

    # Check if we have cameras
    if not config.cameras:
        print("No cameras configured!")
        print()
        print("Options:")
        print("  1. Run setup wizard:  python -m rack_inventory --setup")
        print("  2. Scan for cameras:  python -m rack_inventory --scan")
        print("  3. Add camera by IP:  python -m rack_inventory --camera 192.168.122.128")
        return

    # Run viewer
    print()
    print(f"Starting with {len(config.cameras)} camera(s):")
    for cam in config.cameras:
        print(f"  - {cam.name}: {cam.ip}")
    print()

    if args.realtime:
        print("Using REAL-TIME viewer (multiprocessing)...")
        run_realtime(config)
    else:
        run_viewer(config)


if __name__ == "__main__":
    main()
