#!/usr/bin/env python3
"""
Get a pre-trained cardboard box detection model!
Much faster than training from scratch.
"""

import os
import urllib.request

def download_roboflow_model():
    """Download pre-trained box model from Roboflow."""
    print("=" * 50)
    print("  GET PRE-TRAINED BOX MODEL")
    print("=" * 50)

    # Option 1: Use Roboflow's inference API (free tier)
    print("""
Option 1: Roboflow (Best - trained on 10000+ box images)
---------------------------------------------------------
1. Go to: https://universe.roboflow.com/search?q=cardboard+box
2. Pick a model (e.g., "cardboard-box-detection")
3. Click "Deploy" -> "Download" -> "YOLOv8"
4. Save as box_model.pt in this folder

Top models:
- https://universe.roboflow.com/myworkspace-cvwlg/cardboard-box-segxq
- https://universe.roboflow.com/jacob-solawetz/logistics-sz9jr
""")

    # Option 2: Use YOLO with COCO (has "box" but it's TV/monitor box)
    print("""
Option 2: Fine-tune YOLOv8 (Quick - 5 min setup)
------------------------------------------------
YOLOv8 is pre-trained on COCO which doesn't have "cardboard box"
but we can fine-tune it quickly with your collected data.
""")

    # Option 3: Use GroundingDINO efficiently
    print("""
Option 3: Smart GroundingDINO (What you have - optimized)
----------------------------------------------------------
GroundingDINO already works! We can make it faster:
- Run detection every 2-3 seconds (not every frame)
- Use GPU if available
- Cache results between frames
""")

    # Check what you have
    base_dir = os.path.dirname(__file__)
    training_data = os.path.join(base_dir, "training_data", "images")
    auto_collected = os.path.join(base_dir, "auto_collected")

    total_samples = 0
    if os.path.exists(training_data):
        total_samples += len([f for f in os.listdir(training_data) if f.endswith('.jpg')])
    if os.path.exists(auto_collected):
        total_samples += len([f for f in os.listdir(auto_collected) if f.endswith('.jpg')])

    print(f"""
Your current data: {total_samples} samples
{"Ready to train!" if total_samples >= 20 else f"Need {20 - total_samples} more samples"}

RECOMMENDED APPROACH:
=====================
1. Collect 30-50 samples using count_gdino.py (auto-collect)
2. Review with review_samples.py
3. Train with train_from_dino.py
4. Your model will be optimized for YOUR specific boxes & camera angle!

This is actually BETTER than generic pre-trained because:
- Trained on YOUR camera angle
- Trained on YOUR box types
- Trained on YOUR lighting conditions
""")


def optimize_gdino():
    """Tips to make GroundingDINO faster."""
    print("""
SPEED UP GROUNDINGDINO:
=======================
1. Use GPU: Already auto-detected
2. Reduce detection frequency: Currently 3s, can reduce to 2s
3. Smaller image: Resize before detection
4. Use TensorRT: Convert model for 3x speed (advanced)

For real-time (30fps), YOLO is the only option.
GroundingDINO will always be 1-3 seconds per detection.
""")


if __name__ == "__main__":
    download_roboflow_model()
    optimize_gdino()
