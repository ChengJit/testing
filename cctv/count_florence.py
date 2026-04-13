#!/usr/bin/env python3
"""
Florence-2 Box Counter
=======================

Microsoft's Florence-2 vision model for accurate box detection.

Usage:
    python count_florence.py .129

Install:
    pip install transformers torch timm einops
"""

import sys
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rack_inventory.florence_counter import run_florence_counter

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Florence-2 Box Counter")
        print("=" * 40)
        print()
        print("Microsoft's vision model - very accurate")
        print("for detecting individual objects.")
        print()
        print("Usage: python count_florence.py <camera_ip>")
        print("Example: python count_florence.py .129")
        print()
        print("Install:")
        print("  pip install transformers torch timm einops")
    else:
        run_florence_counter(sys.argv[1])
