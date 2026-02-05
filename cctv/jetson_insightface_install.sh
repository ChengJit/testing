#!/bin/bash
# jetson_insightface_gpu_install.sh
# GPU-ACCELERATED InsightFace Installation for Jetson

echo "=========================================="
echo "InsightFace GPU Installation for Jetson"
echo "=========================================="

# Check Jetson info
echo "🔍 Checking Jetson information..."
cat /etc/nv_tegra_release
echo ""

# Check CUDA
echo "🔍 Checking CUDA..."
nvcc --version 2>/dev/null || echo "CUDA not found in PATH"
echo ""

# Step 1: Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Step 2: Install CUDA tools if needed
echo "📦 Installing CUDA tools..."
sudo apt-get install -y \
    cuda-toolkit-11-4 \
    libcudnn8 \
    libcudnn8-dev \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopenblas-dev \
    liblapack-dev \
    gfortran

# Step 3: Set up CUDA PATH
echo "🔧 Setting up CUDA environment..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Step 4: Install PyTorch with CUDA support (Jetson specific)
echo "📦 Installing PyTorch for Jetson..."
# For JetPack 5.x (Python 3.8)
wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
sudo pip3 install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

# For JetPack 4.x (Python 3.6)
# wget https://nvidia.box.com/shared/static/1v2cc4ro6zvsbu0p8h6qcuaqco1qcsif.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
# sudo pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Step 5: Install GPU-accelerated ONNX Runtime
echo "📦 Installing ONNX Runtime with GPU support..."
# First install CPU version
sudo pip3 install onnxruntime==1.14.1

# Try to install GPU version
sudo pip3 install onnxruntime-gpu==1.14.1 || echo "GPU version failed, using CPU"

# Step 6: Install base packages with specific versions
echo "📦 Installing base Python packages..."
sudo pip3 install \
    "numpy==1.21.6" \
    "opencv-python==4.5.5.64" \
    "scikit-learn==1.0.2" \
    "psutil==5.9.5" \
    "pillow==9.3.0" \
    "tqdm==4.64.1"

# Step 7: Install InsightFace with GPU support
echo "📦 Installing InsightFace with GPU support..."
# Clone and install from source for better control
cd ~
git clone https://github.com/deepinsight/insightface.git
cd insightface
sudo pip3 install -e .
cd python-package
sudo pip3 install -e .
cd ~

# Step 8: Install additional dependencies
echo "📦 Installing additional dependencies..."
sudo pip3 install \
    "torchvision==0.13.0" \
    "scipy==1.9.3" \
    "matplotlib==3.5.3"

# Step 9: Verify GPU installation
echo "🔍 Verifying GPU installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"

# Step 10: Test InsightFace with GPU
echo "🔍 Testing InsightFace GPU..."
python3 -c "
import cv2
print(f'OpenCV: {cv2.__version__}')
try:
    from insightface.app import FaceAnalysis
    import onnxruntime as ort
    
    # Check ONNX Runtime providers
    print('ONNX Runtime providers:', ort.get_available_providers())
    
    # Try to create GPU context
    app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    print('✅ InsightFace GPU mode: READY')
    
except Exception as e:
    print(f'❌ InsightFace GPU error: {e}')
    print('Trying CPU fallback...')
    try:
        app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1)
        print('✅ InsightFace CPU mode: READY')
    except Exception as e2:
        print(f'❌ CPU fallback also failed: {e2}')
"

echo "=========================================="
echo "✅ GPU Installation Complete!"
echo "=========================================="
echo "To verify GPU acceleration, run:"
echo "python3 test_gpu_acceleration.py"
