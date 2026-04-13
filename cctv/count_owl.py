#!/usr/bin/env python3
"""
OWLv2 Box Counter
==================

Google's OWL-ViT v2 - Open-Vocabulary Object Detection
More stable than Florence-2 and works well for box detection.

Usage:
    python count_owl.py .129

Install:
    pip install transformers torch
"""

import sys
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rack_inventory.owl_counter import run_owl_counter

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("OWLv2 Box Counter")
        print("=" * 40)
        print()
        print("Google's OWL-ViT v2 - Open-Vocabulary")
        print("Object Detection. Very accurate for")
        print("finding specific objects by text prompt.")
        print()
        print("Usage: python count_owl.py <camera_ip>")
        print("Example: python count_owl.py .129")
        print()
        print("Install:")
        print("  pip install transformers torch")
    else:
        run_owl_counter(sys.argv[1])
