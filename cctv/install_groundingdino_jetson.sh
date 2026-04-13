#!/bin/bash
# GroundingDINO Installation Script for Jetson Orin Nano (JetPack 6)
# Run: chmod +x install_groundingdino_jetson.sh && ./install_groundingdino_jetson.sh

set -e

echo "=========================================="
echo "  GROUNDINGDINO INSTALLER FOR JETSON"
echo "=========================================="

cd ~/test

# Step 1: Remove old installation
echo ""
echo "[1/6] Removing old GroundingDINO..."
rm -rf GroundingDINO
pip3 uninstall groundingdino -y 2>/dev/null || true

# Step 2: Clone fresh
echo ""
echo "[2/6] Cloning GroundingDINO..."
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO

# Remove pyproject.toml to avoid build isolation
rm -f pyproject.toml

# Step 3: Check PyTorch
echo ""
echo "[3/6] Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Step 4: Install dependencies
echo ""
echo "[4/6] Installing dependencies..."
pip3 install timm==0.9.2 transformers==4.36.0 supervision

# Step 5: Patch and build
echo ""
echo "[5/6] Patching and building..."

# Patch files for PyTorch 2.x
python3 << 'PATCHSCRIPT'
import os

print("Patching files for PyTorch 2.x compatibility...")

# Patch 1: CUDA file - fix deprecated API
f1 = "groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu"
if os.path.exists(f1):
    with open(f1, 'r') as f:
        c = f.read()
    c = c.replace("AT_DISPATCH_FLOATING_TYPES(value.type(),", "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),")
    with open(f1, 'w') as f:
        f.write(c)
    print(f"  Patched: {f1}")

# Patch 2: Header file - fix deprecated API
f2 = "groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn.h"
if os.path.exists(f2):
    with open(f2, 'r') as f:
        c = f.read()
    c = c.replace("value.type().is_cuda()", "value.is_cuda()")
    with open(f2, 'w') as f:
        f.write(c)
    print(f"  Patched: {f2}")

# Patch 3: Force PyTorch implementation (skip CUDA ops that may fail)
f3 = "groundingdino/models/GroundingDINO/ms_deform_attn.py"
if os.path.exists(f3):
    with open(f3, 'r') as f:
        c = f.read()
    c = c.replace("MultiScaleDeformableAttnFunction.apply(", "multi_scale_deformable_attn_pytorch(")
    with open(f3, 'w') as f:
        f.write(c)
    print(f"  Patched: {f3}")

print("Done patching!")
PATCHSCRIPT

# Build
echo ""
echo "Building GroundingDINO..."
export TORCH_CUDA_ARCH_LIST="8.7"
python3 setup.py build install --user

# Step 6: Download weights
echo ""
echo "[6/6] Downloading weights..."
mkdir -p weights
if [ ! -f "weights/groundingdino_swint_ogc.pth" ]; then
    wget -q --show-progress -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "  Weights already exist"
fi

# Test
echo ""
echo "=========================================="
echo "  TESTING INSTALLATION"
echo "=========================================="
python3 -c "
import sys
sys.path.insert(0, '.')
from groundingdino.util.inference import load_model
print('GroundingDINO import: SUCCESS!')
"

echo ""
echo "=========================================="
echo "  INSTALLATION COMPLETE!"
echo "=========================================="
echo ""
echo "Run your app with:"
echo "  cd ~/test"
echo "  python3 cctv/count_scan_place_jetson.py 192.168.122.128"
echo ""
