#!/usr/bin/env python3
"""
Train YOLO from GroundingDINO labels!

Workflow:
1. Run count_gdino.py, press C when detection is correct
2. Collect 20-50 samples
3. Run this script to train YOLO
4. Use count_custom.py with faster, trained model!

GroundingDINO = Teacher (accurate but slow)
YOLO = Student (fast, learns from teacher)
"""

import os
import shutil
import random

def prepare_dataset():
    """Combine training data from all sources."""
    base_dir = os.path.dirname(__file__)

    # Source folders
    sources = [
        os.path.join(base_dir, "training_data"),      # From GroundingDINO (reviewed)
        os.path.join(base_dir, "active_learning"),    # From count_custom.py
        os.path.join(base_dir, "auto_collected"),     # Auto-collected (unreviewed)
    ]

    # Output folder
    dataset_dir = os.path.join(base_dir, "yolo_dataset")
    train_img = os.path.join(dataset_dir, "train", "images")
    train_lbl = os.path.join(dataset_dir, "train", "labels")
    val_img = os.path.join(dataset_dir, "val", "images")
    val_lbl = os.path.join(dataset_dir, "val", "labels")

    # Clean and create
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(train_img)
    os.makedirs(train_lbl)
    os.makedirs(val_img)
    os.makedirs(val_lbl)

    # Collect all samples
    samples = []

    for source in sources:
        if not os.path.exists(source):
            continue

        # Check for images/labels structure
        img_dir = os.path.join(source, "images")
        lbl_dir = os.path.join(source, "labels")

        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            # Structured format
            for img_file in os.listdir(img_dir):
                if img_file.endswith(('.jpg', '.png')):
                    name = os.path.splitext(img_file)[0]
                    lbl_file = name + ".txt"
                    if os.path.exists(os.path.join(lbl_dir, lbl_file)):
                        samples.append({
                            'img': os.path.join(img_dir, img_file),
                            'lbl': os.path.join(lbl_dir, lbl_file)
                        })
        else:
            # Flat format (image and label in same folder)
            for f in os.listdir(source):
                if f.endswith(('.jpg', '.png')):
                    name = os.path.splitext(f)[0]
                    lbl_file = name + ".txt"
                    if os.path.exists(os.path.join(source, lbl_file)):
                        samples.append({
                            'img': os.path.join(source, f),
                            'lbl': os.path.join(source, lbl_file)
                        })

    print(f"Found {len(samples)} labeled samples")

    if len(samples) < 5:
        print("\nNeed at least 5 samples!")
        print("Run count_gdino.py and press C when boxes are detected correctly.")
        return None, 0

    # Shuffle and split 80/20
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    val_samples = samples[split:] if split < len(samples) else samples[-2:]

    # Copy files
    for i, s in enumerate(train_samples):
        ext = os.path.splitext(s['img'])[1]
        shutil.copy(s['img'], os.path.join(train_img, f"img_{i:04d}{ext}"))
        shutil.copy(s['lbl'], os.path.join(train_lbl, f"img_{i:04d}.txt"))

    for i, s in enumerate(val_samples):
        ext = os.path.splitext(s['img'])[1]
        shutil.copy(s['img'], os.path.join(val_img, f"img_{i:04d}{ext}"))
        shutil.copy(s['lbl'], os.path.join(val_lbl, f"img_{i:04d}.txt"))

    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")

    # Create dataset.yaml
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    yaml_content = f"""
path: {dataset_dir}
train: train/images
val: val/images

names:
  0: box
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    return yaml_path, len(samples)


def train_yolo(yaml_path, epochs=50):
    """Train YOLO model."""
    from ultralytics import YOLO

    base_dir = os.path.dirname(__file__)
    output_model = os.path.join(base_dir, "box_model.pt")

    # Start from pretrained YOLOv8
    print("\nLoading YOLOv8 base model...")
    model = YOLO("yolov8n.pt")  # nano = fastest

    print(f"\nTraining for {epochs} epochs...")
    print("This may take 5-30 minutes depending on your hardware.\n")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=8,
        patience=10,
        project=os.path.join(base_dir, "yolo_training"),
        name="box_detector",
        exist_ok=True,
        verbose=True
    )

    # Copy best model
    best_model = os.path.join(base_dir, "yolo_training", "box_detector", "weights", "best.pt")
    if os.path.exists(best_model):
        shutil.copy(best_model, output_model)
        print(f"\n{'='*50}")
        print(f"  MODEL SAVED: {output_model}")
        print(f"{'='*50}")
        print("\nNow run: python count_custom.py .129")
        print("Your YOLO model is now trained on YOUR boxes!")
        return True
    else:
        print("Training failed - no model produced")
        return False


def main():
    print("=" * 60)
    print("  TRAIN YOLO FROM GROUNDINGDINO LABELS")
    print("  (Knowledge Distillation: DINO teaches YOLO)")
    print("=" * 60)

    # Prepare dataset
    print("\n[1] Preparing dataset...")
    yaml_path, num_samples = prepare_dataset()

    if yaml_path is None:
        return

    # Ask for epochs
    print(f"\n[2] Training configuration")
    print(f"    Samples: {num_samples}")

    if num_samples < 20:
        epochs = 30
        print(f"    Few samples -> quick training ({epochs} epochs)")
    elif num_samples < 50:
        epochs = 50
        print(f"    Medium samples -> normal training ({epochs} epochs)")
    else:
        epochs = 100
        print(f"    Many samples -> thorough training ({epochs} epochs)")

    # Train
    print(f"\n[3] Training YOLO...")
    train_yolo(yaml_path, epochs)


if __name__ == "__main__":
    main()
