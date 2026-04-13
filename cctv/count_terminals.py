#!/usr/bin/env python3
"""
Terminal Box Counter
====================

Count terminal boxes on shelf with change detection.

Usage:
    python count_terminals.py .129
"""

import sys
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rack_inventory.terminal_counter import run_terminal_counter

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Terminal Box Counter")
        print("=" * 40)
        print()
        print("Usage: python count_terminals.py <camera_ip>")
        print()
        print("Example:")
        print("  python count_terminals.py .129")
        print()
        print("Setup:")
        print("  1. Draw ROI around terminal box area")
        print("  2. Press B to set baseline count")
        print("  3. System will alert when boxes removed")
    else:
        run_terminal_counter(sys.argv[1])
