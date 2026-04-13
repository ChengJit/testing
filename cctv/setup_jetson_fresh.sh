#!/bin/bash
# Fresh GroundingDINO setup for Jetson Orin Nano
# Run: chmod +x setup_jetson_fresh.sh && ./setup_jetson_fresh.sh

set -e  # Exit on error

echo "=========================================="
echo "  FRESH GROUNDINGDINO SETUP FOR JETSON"
echo "=========================================="

# 1. Remove old installation
echo ""
echo "[1/7] Removing old GroundingDINO..."
pip3 uninstall groundingdino -y 2>/dev/null || true
rm -rf ~/GroundingDINO 2>/dev/null || true
rm -rf ~/test/GroundingDINO 2>/dev/null || true

# 2. Install dependencies
echo ""
echo "[2/7] Installing dependencies..."
pip3 install --upgrade pip
pip3 install ninja numpy pillow opencv-python
pip3 install timm==0.6.13
pip3 install transformers==4.36.0
pip3 install supervision

# 3. Clone fresh GroundingDINO
echo ""
echo "[3/7] Cloning GroundingDINO..."
cd ~/test
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO

# 4. Patch the CUDA files for PyTorch 2.x compatibility
echo ""
echo "[4/7] Patching CUDA files for PyTorch 2.x..."

# Patch ms_deform_attn_cuda.cu - fix deprecated value.type()
CUDA_FILE="groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu"
if [ -f "$CUDA_FILE" ]; then
    cp "$CUDA_FILE" "${CUDA_FILE}.bak"
    sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' "$CUDA_FILE"
    echo "  Patched: $CUDA_FILE"
fi

# Patch ms_deform_attn.h - fix deprecated value.type().is_cuda()
HEADER_FILE="groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn.h"
if [ -f "$HEADER_FILE" ]; then
    cp "$HEADER_FILE" "${HEADER_FILE}.bak"
    sed -i 's/value\.type()\.is_cuda()/value.is_cuda()/g' "$HEADER_FILE"
    echo "  Patched: $HEADER_FILE"
fi

# Patch swin_transformer.py - fix timm import
SWIN_FILE="groundingdino/models/GroundingDINO/backbone/swin_transformer.py"
if [ -f "$SWIN_FILE" ]; then
    cp "$SWIN_FILE" "${SWIN_FILE}.bak"
    sed -i 's/from timm\.models\.layers import/from timm.layers import/g' "$SWIN_FILE"
    echo "  Patched: $SWIN_FILE"
fi

# 5. Build and install
echo ""
echo "[5/7] Building GroundingDINO (this may take a few minutes)..."
export TORCH_CUDA_ARCH_LIST="8.7"  # Jetson Orin architecture
pip3 install -e .

# 6. Download weights
echo ""
echo "[6/7] Downloading model weights..."
mkdir -p weights
cd weights
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "  Weights already exist"
fi
cd ..

# 7. Test installation
echo ""
echo "[7/7] Testing installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

from groundingdino.util.inference import load_model
print('GroundingDINO import: OK')
"

echo ""
echo "=========================================="
echo "  SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Now run:"
echo "  cd ~/test"
echo "  python3 cctv/count_scan_place_jetson.py 192.168.122.128"
echo ""
