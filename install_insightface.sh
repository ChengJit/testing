#!/bin/bash
echo "=========================================="
echo "InsightFace Installation for Jetson"
echo "=========================================="

echo "Updating package list..."
sudo apt-get update

echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopenblas-dev \
    liblapack-dev

echo "Fixing numpy installation..."
sudo pip3 uninstall -y numpy
sudo pip3 install "numpy==1.19.5"

echo "Installing Python dependencies..."
sudo pip3 install \
    "scikit-learn==1.0.2" \
    "psutil==5.9.5" \
    "onnxruntime==1.14.1" \
    "opencv-python==4.5.5.64"

echo "Installing InsightFace..."
sudo pip3 install "insightface==0.7.3"

echo "Testing installation..."
python3 -c "
import numpy as np
print(f'NumPy: {np.__version__}')
import cv2
print(f'OpenCV: {cv2.__version__}')
try:
    import insightface
    print('InsightFace: OK')
except Exception as e:
    print(f'InsightFace error: {e}')
"

echo "=========================================="
echo "Installation complete!"
echo "Now you can run: python3 retrain_embeddings.py"
echo "=========================================="
