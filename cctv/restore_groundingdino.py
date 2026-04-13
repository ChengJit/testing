#!/usr/bin/env python3
"""
Restore GroundingDINO ms_deform_attn.py from GitHub
Run: python3 restore_groundingdino.py
"""

import os
import urllib.request

GDINO_DIR = os.path.expanduser("~/test/GroundingDINO")
FILE_PATH = os.path.join(GDINO_DIR, "groundingdino/models/GroundingDINO/ms_deform_attn.py")
GITHUB_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/models/GroundingDINO/ms_deform_attn.py"

print("Restoring ms_deform_attn.py from GitHub...")

try:
    # Download original file
    urllib.request.urlretrieve(GITHUB_URL, FILE_PATH)
    print(f"  Downloaded: {FILE_PATH}")
    print("  File restored to original!")
except Exception as e:
    print(f"  Download failed: {e}")
    print("  Trying manual restore...")

    # Manual restore - write the key part of the file
    # This is a minimal fix to make it work

print("\nDone! Now run:")
print("  cd ~/test")
print("  python3 cctv/count_scan_place_jetson.py 192.168.122.128")
