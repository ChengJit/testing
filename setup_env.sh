# Remove old venv
deactivate
rm -rf ~/pytorch-venv

# Recreate venv
python3 -m venv ~/pytorch-venv
source ~/pytorch-venv/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install compatible NumPy
python3 -m pip install numpy==1.26.1

# Install NVIDIA Jetson GPU PyTorch + torchvision
python3 -m pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-cache-dir torch torchvision

# Install InsightFace
python3 -m pip install insightface

# For JetPack 6 with CUDA 12.6 support
pip3 install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
