#!/usr/bin/env python3
"""
YOLOv8 Training Script for Custom Box Detection

Prerequisites:
1. Collected images in dataset/images/
2. Labels in dataset/labels/ (YOLO format)
3. dataset.yaml config file

Run: python train_yolov8.py
"""

import os

# ============ CONFIGURATION ============
CONFIG = {
    "model": "yolov8n.pt",       # Base model (n=nano, s=small, m=medium)
    "data": "dataset.yaml",      # Dataset config
    "epochs": 50,                # Training epochs
    "imgsz": 480,                # Image size (reduced for Jetson memory)
    "batch": 4,                  # Batch size (reduced for Jetson Orin Nano)
    "device": 0,                 # GPU device (0) or "cpu"
    "workers": 2,                # Data loader workers (reduced)
    "patience": 10,              # Early stopping patience
}
# =======================================


def create_dataset_yaml():
    """Create dataset.yaml if it doesn't exist."""
    yaml_path = "dataset.yaml"

    if os.path.exists(yaml_path):
        print(f"  Found existing {yaml_path}")
        return yaml_path

    content = """# YOLOv8 Dataset Configuration
path: dataset
train: images
val: images

names:
  0: box
"""

    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"  Created {yaml_path}")
    return yaml_path


def check_dataset():
    """Check if dataset is ready."""
    images_dir = "dataset/images"
    labels_dir = "dataset/labels"

    if not os.path.exists(images_dir):
        print(f"ERROR: {images_dir} not found!")
        print("Run collect_training_images.py first")
        return False

    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"  Found {len(images)} images")

    if len(images) < 10:
        print("WARNING: Very few images. Recommend at least 50-100 for good results.")

    if not os.path.exists(labels_dir):
        print(f"\nERROR: {labels_dir} not found!")
        print("You need to label your images first:")
        print("  1. pip install labelImg")
        print("  2. labelImg dataset/images dataset/labels")
        print("  3. Draw boxes around each box and save")
        return False

    labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(f"  Found {len(labels)} label files")

    if len(labels) == 0:
        print("\nERROR: No label files found!")
        print("Label your images with LabelImg first.")
        return False

    if len(labels) < len(images) * 0.8:
        print(f"WARNING: Only {len(labels)}/{len(images)} images are labeled")

    return True


def train():
    """Train YOLOv8 model."""
    print("=" * 50)
    print("  YOLOv8 TRAINING")
    print("=" * 50)

    # Check dataset
    print("\nChecking dataset...")
    if not check_dataset():
        return

    # Create dataset.yaml
    print("\nPreparing config...")
    data_yaml = create_dataset_yaml()

    # Import and train
    print("\nStarting training...")
    print(f"  Base model: {CONFIG['model']}")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Image size: {CONFIG['imgsz']}")
    print(f"  Batch size: {CONFIG['batch']}")
    print("")

    try:
        from ultralytics import YOLO

        # Load base model
        model = YOLO(CONFIG["model"])

        # Train
        results = model.train(
            data=data_yaml,
            epochs=CONFIG["epochs"],
            imgsz=CONFIG["imgsz"],
            batch=CONFIG["batch"],
            device=CONFIG["device"],
            workers=CONFIG["workers"],
            patience=CONFIG["patience"],
            save=True,
            plots=True,
            verbose=True,
        )

        print("\n" + "=" * 50)
        print("  TRAINING COMPLETE")
        print("=" * 50)
        print(f"\n  Best model saved to:")
        print(f"    runs/detect/train/weights/best.pt")
        print(f"\n  To use in your detector:")
        print(f'    CONFIG["model_path"] = "runs/detect/train/weights/best.pt"')
        print("=" * 50)

    except ImportError:
        print("ERROR: ultralytics not installed!")
        print("Install with: pip install ultralytics")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train()
