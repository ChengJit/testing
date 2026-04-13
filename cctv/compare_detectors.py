#!/usr/bin/env python3
"""
Compare Box Detection Methods
==============================

Compare different detection methods for your terminal boxes:
- Edge Detection (fast, traditional CV)
- Color Segmentation (fast, for brown cardboard)
- Hybrid (edge + color combined)
- YOLO-World (AI, zero-shot)
- GroundingDINO (AI, more accurate)

Usage:
    python compare_detectors.py .129
"""

import sys
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rack_inventory.box_detector_cv import compare_methods

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Compare Box Detection Methods")
        print("=" * 40)
        print()
        print("Usage: python compare_detectors.py <camera_ip>")
        print("Example: python compare_detectors.py .129")
        print()
        print("Controls:")
        print("  1 - Edge Detection (fast)")
        print("  2 - Color Segmentation (fast)")
        print("  3 - Hybrid (recommended)")
        print("  4 - YOLO-World (AI)")
        print("  5 - GroundingDINO (most accurate)")
        print("  A - Compare ALL methods side-by-side")
        print("  R - Draw ROI")
        print("  S - Screenshot")
        print("  Q - Quit")
    else:
        compare_methods(sys.argv[1])
