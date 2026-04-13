#!/usr/bin/env python3
"""
Train YOLO on YOUR box data!
Uses auto-labeled data from GroundingDINO.
"""

import os
import shutil
import random

def prepare_dataset():
    """Split data into train/val and create config."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "training_data")
    img_dir = os.path.join(data_dir, "images")
    lbl_dir = os.path.join(data_dir, "labels")

    # Check data exists
    if not os.path.exists(img_dir):
        print("No training data found!")
        print("Run count_gdino.py and press C to collect data first.")
        return None

    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    print(f"Found {len(images)} training images!")

    if len(images) < 20:
        print("Need at least 20 images. Collect more!")
        return None

    # Create dataset structure
    dataset_dir = os.path.join(base_dir, "box_dataset")
    train_img = os.path.join(dataset_dir, "train", "images")
    train_lbl = os.path.join(dataset_dir, "train", "labels")
    val_img = os.path.join(dataset_dir, "val", "images")
    val_lbl = os.path.join(dataset_dir, "val", "labels")

    for d in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(d, exist_ok=True)

    # Shuffle and split 80/20
    random.shuffle(images)
    split = int(len(images) * 0.8)
    train_files = images[:split]
    val_files = images[split:]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Copy files
    for f in train_files:
        shutil.copy(os.path.join(img_dir, f), os.path.join(train_img, f))
        lbl_f = f.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(lbl_dir, lbl_f)):
            shutil.copy(os.path.join(lbl_dir, lbl_f), os.path.join(train_lbl, lbl_f))

    for f in val_files:
        shutil.copy(os.path.join(img_dir, f), os.path.join(val_img, f))
        lbl_f = f.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(lbl_dir, lbl_f)):
            shutil.copy(os.path.join(lbl_dir, lbl_f), os.path.join(val_lbl, lbl_f))

    # Create data.yaml
    yaml_content = f"""# Box Detection Dataset
path: {dataset_dir}
train: train/images
val: val/images

names:
  0: box
"""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Dataset ready at: {dataset_dir}")
    return yaml_path


def train():
    """Train YOLO model."""
    print("=" * 50)
    print("  YOLO BOX TRAINER")
    print("=" * 50)

    # Prepare dataset
    yaml_path = prepare_dataset()
    if not yaml_path:
        return

    # Install ultralytics if needed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nInstalling ultralytics...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO

    print("\nStarting training...")
    print("This may take 10-30 minutes depending on your hardware.\n")

    # Use YOLOv8 MEDIUM for better accuracy!
    model = YOLO("yolov8m.pt")

    # Train harder!
    results = model.train(
        data=yaml_path,
        epochs=100,          # More epochs = smarter
        imgsz=640,           # Image size
        batch=4,             # Smaller batch for bigger model
        name="box_detector",
        patience=20,         # More patience
        augment=True,        # Data augmentation
        mosaic=1.0,          # Mosaic augmentation
        mixup=0.1,           # Mixup augmentation
        verbose=True
    )

    # Find best model
    best_model = os.path.join("runs", "detect", "box_detector", "weights", "best.pt")
    if os.path.exists(best_model):
        # Copy to easy location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        final_model = os.path.join(base_dir, "box_model.pt")
        shutil.copy(best_model, final_model)
        print("\n" + "=" * 50)
        print("  TRAINING COMPLETE!")
        print("=" * 50)
        print(f"\nModel saved to: {final_model}")
        print("\nTo use your model, run:")
        print("  python count_custom.py .129")
    else:
        print("Training may have failed. Check errors above.")


if __name__ == "__main__":
    train()
