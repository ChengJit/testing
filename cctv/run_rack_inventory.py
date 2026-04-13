#!/usr/bin/env python3
"""
Rack Inventory Scanner - Quick Start
=====================================

Run this script to scan for cameras and view rack inventory.

Usage:
    python run_rack_inventory.py              # Auto-scan and view
    python run_rack_inventory.py --scan       # Just scan for cameras
    python run_rack_inventory.py --setup      # Interactive setup
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rack_inventory.app import main

if __name__ == "__main__":
    main()
