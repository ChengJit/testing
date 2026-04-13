#!/usr/bin/env python3
"""
Fix GroundingDINO to use compiled CUDA ops
Run: python3 fix_groundingdino.py
"""

import os

GDINO_DIR = os.path.expanduser("~/test/GroundingDINO")

print("Fixing GroundingDINO to use CUDA ops...")

# Fix ms_deform_attn.py - revert to CUDA version
filepath = os.path.join(GDINO_DIR, "groundingdino/models/GroundingDINO/ms_deform_attn.py")

if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Revert PyTorch fallback to CUDA ops
    if "multi_scale_deformable_attn_pytorch(" in content:
        content = content.replace(
            "multi_scale_deformable_attn_pytorch(",
            "MultiScaleDeformableAttnFunction.apply("
        )
        with open(filepath, 'w') as f:
            f.write(content)
        print("  Reverted to CUDA ops: MultiScaleDeformableAttnFunction.apply()")
    else:
        print("  Already using CUDA ops")
else:
    print(f"  File not found: {filepath}")

print("\nDone! Now run:")
print("  cd ~/test")
print("  python3 cctv/count_scan_place_jetson.py 192.168.122.128")
