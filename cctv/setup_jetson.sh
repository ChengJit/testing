#!/bin/bash
# Setup script for Jetson Orin Nano
# Run: chmod +x setup_jetson.sh && ./setup_jetson.sh

echo "=========================================="
echo "  JETSON ORIN NANO SETUP"
echo "=========================================="

# Check if running on Jetson
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    echo "Device: $MODEL"
else
    echo "Warning: Not running on Jetson"
fi

# Check CUDA
echo ""
echo "Checking CUDA..."
if command -v nvcc &> /dev/null; then
    nvcc --version | head -4
else
    echo "CUDA not found in PATH"
    echo "Try: export PATH=/usr/local/cuda/bin:\$PATH"
fi

# Check Python
echo ""
echo "Checking Python..."
python3 --version

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly

# Install Python packages
echo ""
echo "Installing Python packages..."
pip3 install --upgrade pip
pip3 install numpy pillow

# Check if PyTorch is installed (Jetson uses special wheels)
echo ""
echo "Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "PyTorch not installed!"
    echo ""
    echo "For Jetson Orin Nano, install PyTorch from NVIDIA:"
    echo "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    echo ""
    echo "Example for JetPack 5.x:"
    echo "  wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O torch-1.11.0-cp38-cp38-linux_aarch64.whl"
    echo "  pip3 install torch-1.11.0-cp38-cp38-linux_aarch64.whl"
}

# Install GroundingDINO
echo ""
echo "Setting up GroundingDINO..."
GDINO_DIR="$HOME/GroundingDINO"

if [ -d "$GDINO_DIR" ]; then
    echo "GroundingDINO already exists at $GDINO_DIR"
else
    echo "Cloning GroundingDINO..."
    cd $HOME
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO
    pip3 install -e .
fi

# Download weights
WEIGHTS_DIR="$GDINO_DIR/weights"
WEIGHTS_FILE="$WEIGHTS_DIR/groundingdino_swint_ogc.pth"

if [ -f "$WEIGHTS_FILE" ]; then
    echo "Weights already downloaded"
else
    echo "Downloading GroundingDINO weights..."
    mkdir -p "$WEIGHTS_DIR"
    wget -P "$WEIGHTS_DIR" https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi

# Install transformers (specific version for compatibility)
echo ""
echo "Installing transformers..."
pip3 install transformers==4.36.0

# Test import
echo ""
echo "Testing imports..."
python3 -c "
import cv2
print(f'OpenCV: {cv2.__version__}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" || echo "Some imports failed - check errors above"

echo ""
echo "=========================================="
echo "  SETUP COMPLETE"
echo "=========================================="
echo ""
echo "To run the inventory system:"
echo "  python3 count_scan_place_jetson.py 192.168.122.128"
echo ""
echo "Edit CONFIG at the top of the file to adjust settings."
