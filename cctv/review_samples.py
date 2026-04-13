#!/usr/bin/env python3
"""
Review auto-collected training samples.
Approve good ones, reject bad ones.

Controls:
  Y / SPACE = Approve (move to training data)
  N / DELETE = Reject (delete)
  E = Edit count (if wrong)
  ← → = Navigate
  Q = Quit
"""

import cv2
import os
import shutil

def load_samples(collect_dir):
    """Load all auto-collected samples."""
    samples = []

    if not os.path.exists(collect_dir):
        return samples

    for f in os.listdir(collect_dir):
        if f.endswith('.jpg'):
            name = os.path.splitext(f)[0]
            lbl_file = name + ".txt"
            lbl_path = os.path.join(collect_dir, lbl_file)

            samples.append({
                'img_path': os.path.join(collect_dir, f),
                'lbl_path': lbl_path if os.path.exists(lbl_path) else None,
                'name': name
            })

    # Sort by timestamp
    samples.sort(key=lambda x: x['name'])
    return samples


def load_labels(lbl_path):
    """Load YOLO labels."""
    if not lbl_path or not os.path.exists(lbl_path):
        return []

    boxes = []
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, cx, cy, w, h = map(float, parts[:5])
                boxes.append((cx, cy, w, h))
    return boxes


def draw_sample(img, boxes, index, total, status=""):
    """Draw sample with boxes."""
    display = img.copy()
    h, w = display.shape[:2]

    # Draw boxes
    for i, (cx, cy, bw, bh) in enumerate(boxes):
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, str(i+1), (x1+5, y1+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Info bar
    cv2.rectangle(display, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(display, f"Sample {index+1}/{total}  |  Boxes: {len(boxes)}",
               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, status, (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Controls
    cv2.rectangle(display, (0, h-30), (w, h), (40, 40, 40), -1)
    cv2.putText(display, "Y:Approve  N:Reject  E:Edit  <-/->:Navigate  Q:Quit",
               (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return display


def approve_sample(sample, training_dir):
    """Move approved sample to training data."""
    img_dir = os.path.join(training_dir, "images")
    lbl_dir = os.path.join(training_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Count existing
    existing = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    new_name = f"approved_{existing:04d}"

    # Move files
    shutil.move(sample['img_path'], os.path.join(img_dir, new_name + ".jpg"))
    if sample['lbl_path'] and os.path.exists(sample['lbl_path']):
        shutil.move(sample['lbl_path'], os.path.join(lbl_dir, new_name + ".txt"))

    return existing + 1


def reject_sample(sample):
    """Delete rejected sample."""
    if os.path.exists(sample['img_path']):
        os.remove(sample['img_path'])
    if sample['lbl_path'] and os.path.exists(sample['lbl_path']):
        os.remove(sample['lbl_path'])


def main():
    base_dir = os.path.dirname(__file__)
    collect_dir = os.path.join(base_dir, "auto_collected")
    training_dir = os.path.join(base_dir, "training_data")

    print("=" * 50)
    print("  REVIEW AUTO-COLLECTED SAMPLES")
    print("=" * 50)

    samples = load_samples(collect_dir)

    if not samples:
        print(f"\nNo samples to review in: {collect_dir}")
        print("Run count_gdino.py with auto-collect enabled first.")
        return

    print(f"\nFound {len(samples)} samples to review")
    print("\nControls:")
    print("  Y/SPACE = Approve (move to training)")
    print("  N/DEL   = Reject (delete)")
    print("  ←/→     = Navigate")
    print("  Q       = Quit")

    cv2.namedWindow("Review", cv2.WINDOW_NORMAL)

    idx = 0
    approved = 0
    rejected = 0

    while samples and idx < len(samples):
        sample = samples[idx]

        # Load image and labels
        img = cv2.imread(sample['img_path'])
        if img is None:
            samples.pop(idx)
            continue

        boxes = load_labels(sample['lbl_path'])

        status = f"Approved: {approved}  Rejected: {rejected}  Remaining: {len(samples)}"
        display = draw_sample(img, boxes, idx, len(samples), status)

        # Resize for display
        dh, dw = display.shape[:2]
        scale = min(1280/dw, 720/dh, 1.0)
        if scale < 1.0:
            display = cv2.resize(display, (int(dw*scale), int(dh*scale)))

        cv2.imshow("Review", display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('y') or key == 32:  # Y or SPACE
            total = approve_sample(sample, training_dir)
            print(f"  Approved! (total: {total})")
            approved += 1
            samples.pop(idx)
            if idx >= len(samples):
                idx = max(0, len(samples) - 1)
        elif key == ord('n') or key == 255:  # N or DELETE
            reject_sample(sample)
            print(f"  Rejected")
            rejected += 1
            samples.pop(idx)
            if idx >= len(samples):
                idx = max(0, len(samples) - 1)
        elif key == 81 or key == 2:  # LEFT arrow
            idx = max(0, idx - 1)
        elif key == 83 or key == 3:  # RIGHT arrow
            idx = min(len(samples) - 1, idx + 1)

    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"  REVIEW COMPLETE")
    print(f"  Approved: {approved}")
    print(f"  Rejected: {rejected}")
    print(f"  Remaining: {len(samples)}")
    print(f"{'='*50}")

    if approved > 0:
        print(f"\nTo train: python train_from_dino.py")


if __name__ == "__main__":
    main()
