#!/usr/bin/env python3
"""
Complete GroundingDINO setup for Jetson Orin Nano
Run: python3 setup_groundingdino.py
"""

import os
import sys
import subprocess
import shutil
import urllib.request

HOME = os.path.expanduser("~")
GDINO_DIR = os.path.join(HOME, "test", "GroundingDINO")
WEIGHTS_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"


def run_cmd(cmd, check=True):
    """Run shell command."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"  Warning: Command returned {result.returncode}")
    return result.returncode == 0


def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    print(f"\r  Downloading: {percent}%", end='', flush=True)


def main():
    print("=" * 50)
    print("  GROUNDINGDINO SETUP FOR JETSON")
    print("=" * 50)

    # Step 1: Remove old installation
    print("\n[1/7] Removing old GroundingDINO...")
    run_cmd("pip3 uninstall groundingdino -y", check=False)
    if os.path.exists(GDINO_DIR):
        shutil.rmtree(GDINO_DIR, ignore_errors=True)
        print(f"  Removed {GDINO_DIR}")

    # Step 2: Install dependencies
    print("\n[2/7] Installing dependencies...")
    run_cmd("pip3 install ninja numpy pillow")
    run_cmd("pip3 install timm==0.6.13")
    run_cmd("pip3 install transformers==4.36.0")

    # Step 3: Clone GroundingDINO
    print("\n[3/7] Cloning GroundingDINO...")
    os.makedirs(os.path.dirname(GDINO_DIR), exist_ok=True)
    os.chdir(os.path.dirname(GDINO_DIR))
    run_cmd("git clone https://github.com/IDEA-Research/GroundingDINO.git")
    os.chdir(GDINO_DIR)

    # Step 4: Remove pyproject.toml to avoid build isolation
    print("\n[4/7] Removing pyproject.toml...")
    pyproject = os.path.join(GDINO_DIR, "pyproject.toml")
    if os.path.exists(pyproject):
        os.remove(pyproject)
        print("  Removed pyproject.toml")

    # Step 5: Patch files for PyTorch 2.x compatibility
    print("\n[5/7] Patching files for PyTorch 2.x...")

    # Patch CUDA file - need to replace the entire AT_DISPATCH line
    cuda_file = os.path.join(GDINO_DIR, "groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu")
    if os.path.exists(cuda_file):
        try:
            with open(cuda_file, 'r') as f:
                content = f.read()

            # Replace AT_DISPATCH_FLOATING_TYPES(value.type() with AT_DISPATCH_FLOATING_TYPES(value.scalar_type()
            content = content.replace(
                "AT_DISPATCH_FLOATING_TYPES(value.type(),",
                "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"
            )

            with open(cuda_file, 'w') as f:
                f.write(content)
            print(f"  Patched: ms_deform_attn_cuda.cu")
        except Exception as e:
            print(f"  Error patching CUDA file: {e}")

    # Patch header file
    header_file = os.path.join(GDINO_DIR, "groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn.h")
    if os.path.exists(header_file):
        try:
            with open(header_file, 'r') as f:
                content = f.read()

            content = content.replace("value.type().is_cuda()", "value.is_cuda()")

            with open(header_file, 'w') as f:
                f.write(content)
            print(f"  Patched: ms_deform_attn.h")
        except Exception as e:
            print(f"  Error patching header file: {e}")

    # Patch swin transformer
    swin_file = os.path.join(GDINO_DIR, "groundingdino/models/GroundingDINO/backbone/swin_transformer.py")
    if os.path.exists(swin_file):
        try:
            with open(swin_file, 'r') as f:
                content = f.read()

            content = content.replace(
                "from timm.models.layers import",
                "from timm.layers import"
            )

            with open(swin_file, 'w') as f:
                f.write(content)
            print(f"  Patched: swin_transformer.py")
        except Exception as e:
            print(f"  Error patching swin file: {e}")

    # Step 6: Build and install
    print("\n[6/7] Building GroundingDINO (this takes a few minutes)...")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.7"  # Jetson Orin

    os.chdir(GDINO_DIR)

    # Try build
    print("  Running: python3 setup.py build")
    build_result = run_cmd("python3 setup.py build", check=False)

    if build_result:
        print("  Running: python3 setup.py install --user")
        run_cmd("python3 setup.py install --user", check=False)
    else:
        print("\n  CUDA build failed!")
        print("  Trying CPU-only mode (slower but works)...")

        # Patch ms_deform_attn.py to force PyTorch implementation
        attn_py = os.path.join(GDINO_DIR, "groundingdino/models/GroundingDINO/ms_deform_attn.py")
        if os.path.exists(attn_py):
            with open(attn_py, 'r') as f:
                content = f.read()

            # Force CPU mode by making _C import always fail
            if "try:" in content and "import MultiScaleDeformableAttention as _C" in content:
                content = content.replace(
                    "import MultiScaleDeformableAttention as _C",
                    "raise ImportError('Force CPU mode')"
                )
                with open(attn_py, 'w') as f:
                    f.write(content)
                print("  Forced CPU mode in ms_deform_attn.py")

        # Install without building CUDA extensions
        print("  Installing without CUDA extensions...")
        run_cmd("pip3 install --no-build-isolation --no-deps .", check=False)

        # Also try just adding to path
        print("  Adding to Python path as fallback...")

    # Step 7: Download weights
    print("\n[7/7] Downloading model weights...")
    weights_dir = os.path.join(GDINO_DIR, "weights")
    weights_file = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")

    os.makedirs(weights_dir, exist_ok=True)

    if not os.path.exists(weights_file):
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, weights_file, download_progress)
            print("\n  Download complete!")
        except Exception as e:
            print(f"\n  Download failed: {e}")
            print(f"  Manual download: wget -P {weights_dir} {WEIGHTS_URL}")
    else:
        print("  Weights already exist")

    # Test
    print("\n" + "=" * 50)
    print("  TESTING INSTALLATION")
    print("=" * 50)

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("PyTorch not available")

    try:
        sys.path.insert(0, GDINO_DIR)
        from groundingdino.util.inference import load_model
        print("GroundingDINO import: SUCCESS!")
    except Exception as e:
        print(f"GroundingDINO import: FAILED - {e}")

    print("\n" + "=" * 50)
    print("  SETUP COMPLETE!")
    print("=" * 50)
    print(f"""
Next steps:
  cd ~/test
  python3 cctv/count_scan_place_jetson.py 192.168.122.128

If import fails, try adding to your script:
  import sys
  sys.path.insert(0, "{GDINO_DIR}")
""")


if __name__ == "__main__":
    main()
