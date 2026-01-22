#!/usr/bin/env python3
"""
Inventory Door Monitor - Entry Point
=====================================

Monitors people entering/exiting an inventory room and counts boxes they carry.
Optimized for Jetson Orin Nano.

Usage:
    python run_monitor.py                    # Use default config
    python run_monitor.py -s rtsp://...      # Specify camera source
    python run_monitor.py --headless         # Run without display
    python run_monitor.py --setup            # Interactive setup wizard

Controls (when display is shown):
    Q - Quit
    Z - Toggle zone visualization
    S - Toggle statistics overlay
    R - Register unknown face
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inventory_monitor.app import main, InventoryMonitor
from inventory_monitor.config import Config
from inventory_monitor.utils import get_gpu_info
from inventory_monitor.detectors import FaceRecognizer


def setup_wizard():
    """Interactive setup wizard for first-time configuration."""
    print("\n" + "=" * 50)
    print("  Inventory Door Monitor - Setup Wizard")
    print("=" * 50 + "\n")

    # Check GPU
    gpu_info = get_gpu_info()
    print(f"Platform: {gpu_info['platform']}")
    print(f"GPU Available: {gpu_info['gpu_available']}")
    if gpu_info.get('device_name'):
        print(f"GPU Device: {gpu_info['device_name']}")
    print()

    config = Config()

    # Camera source
    print("Step 1: Camera Configuration")
    print("-" * 30)
    source = input(f"Enter camera source (RTSP URL or device ID) [{config.camera.source}]: ").strip()
    if source:
        config.camera.source = source

    # Door position
    print("\nStep 2: Door Position")
    print("-" * 30)
    print("The door line is where entry/exit is detected.")
    print("0.0 = top of frame, 1.0 = bottom of frame")
    door_line = input(f"Door line position (0.0-1.0) [{config.zone.door_line}]: ").strip()
    if door_line:
        try:
            config.zone.door_line = float(door_line)
        except ValueError:
            print("Invalid value, using default")

    # Direction
    print("\nStep 3: Entry Direction")
    print("-" * 30)
    print("Which direction do people move when ENTERING the room?")
    print("1. Moving DOWN in frame (towards bottom)")
    print("2. Moving UP in frame (towards top)")
    direction = input("Choose (1 or 2) [1]: ").strip()
    config.zone.enter_direction_down = direction != "2"

    # TensorRT
    if gpu_info['tensorrt_available']:
        print("\nStep 4: TensorRT Acceleration")
        print("-" * 30)
        use_trt = input("Enable TensorRT acceleration? (y/n) [y]: ").strip().lower()
        config.jetson.use_tensorrt = use_trt != "n"

    # Save config
    config.save("config.json")
    print("\n" + "=" * 50)
    print("  Configuration saved to config.json")
    print("  Run 'python run_monitor.py' to start monitoring")
    print("=" * 50 + "\n")


def train_faces():
    """Train face recognition from images in known_faces directory."""
    print("\n" + "=" * 50)
    print("  Face Training from Images")
    print("=" * 50 + "\n")

    config = Config.load("config.json")

    print(f"Looking for face images in: {config.faces_dir}")
    print()

    # Initialize face recognizer
    recognizer = FaceRecognizer(
        model_name=config.detection.face_model,
        det_size=config.detection.face_det_size,
        faces_dir=str(config.faces_dir),
    )

    # Train from images
    print("Training faces from images...")
    count = recognizer.train_from_images()

    if count > 0:
        print(f"\nSuccessfully trained {count} people!")
        print(f"Embeddings saved to: known_embeddings.npz")
    else:
        print("\nNo faces were trained.")
        print("Make sure you have folders like:")
        print("  known_faces/")
        print("    person_name/")
        print("      image1.jpg")
        print("      image2.jpg")

    print()


def show_status():
    """Show system status and configuration."""
    print("\n" + "=" * 50)
    print("  Inventory Door Monitor - System Status")
    print("=" * 50 + "\n")

    # GPU info
    gpu_info = get_gpu_info()
    print("Hardware:")
    print(f"  Platform: {gpu_info['platform']}")
    print(f"  CUDA: {'Available' if gpu_info['cuda_available'] else 'Not available'}")
    print(f"  TensorRT: {'Available' if gpu_info['tensorrt_available'] else 'Not available'}")
    if gpu_info.get('device_name'):
        print(f"  GPU: {gpu_info['device_name']}")
    if gpu_info.get('gpu_memory_mb'):
        print(f"  GPU Memory: {gpu_info['gpu_memory_mb']} MB")

    # Config
    print("\nConfiguration:")
    try:
        config = Config.load("config.json")
        print(f"  Camera: {config.camera.source}")
        print(f"  Door Line: {config.zone.door_line}")
        print(f"  Entry Direction: {'Down' if config.zone.enter_direction_down else 'Up'}")
        print(f"  TensorRT: {'Enabled' if config.jetson.use_tensorrt else 'Disabled'}")
        print(f"  Processing FPS: {config.jetson.process_fps}")
    except Exception as e:
        print(f"  Could not load config: {e}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inventory Door Monitor - Track people and boxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--config", "-c", default="config.json",
                        help="Path to config file")
    parser.add_argument("--source", "-s",
                        help="Video source (RTSP URL or device ID)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display window")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--setup", action="store_true",
                        help="Run interactive setup wizard")
    parser.add_argument("--status", action="store_true",
                        help="Show system status and exit")
    parser.add_argument("--train", action="store_true",
                        help="Train face recognition from known_faces images")

    args = parser.parse_args()

    if args.setup:
        setup_wizard()
    elif args.status:
        show_status()
    elif args.train:
        train_faces()
    else:
        # Run the main application
        main()
