import torch
import sys
import os

print("=" * 60)
print("PyTorch CUDA Compatibility Check")
print("=" * 60)

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch built with CUDA: {torch.cuda.is_built()}")

# 检查编译配置
print("\nPyTorch build configuration:")
config = torch.__config__.show()
for line in config.split('\n'):
    if 'CUDA' in line or 'cuda' in line:
        print(f"  {line}")

print("\nCUDA availability check:")
try:
    if torch.cuda.is_available():
        print("  ✓ CUDA is available!")
        print(f"    CUDA version: {torch.version.cuda}")
        print(f"    GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # 测试 CUDA 操作
        print("\n  Testing CUDA operations...")
        x = torch.tensor([1.0, 2.0, 3.0])
        if torch.cuda.is_available():
            x = x.cuda()
            print(f"    Tensor moved to CUDA: {x.device}")
            y = x * 2
            print(f"    CUDA operation successful: {y}")
    else:
        print("  ✗ CUDA is NOT available")
        
        # 诊断原因
        print("\n  Troubleshooting:")
        print("  1. Checking CUDA paths...")
        cuda_path = '/usr/local/cuda'
        if os.path.exists(cuda_path):
            print(f"    ✓ CUDA path exists: {cuda_path}")
        else:
            print(f"    ✗ CUDA path not found: {cuda_path}")
            
        print("\n  2. Checking CUDA libraries...")
        libs_to_check = ['libcuda.so', 'libcudart.so', 'libcudnn.so']
        for lib in libs_to_check:
            lib_path = f'/usr/lib/aarch64-linux-gnu/{lib}'
            if os.path.exists(lib_path):
                print(f"    ✓ {lib} found at {lib_path}")
            else:
                # 在其他位置查找
                found = False
                for search_path in ['/usr/local/cuda/lib64', '/usr/lib', '/lib']:
                    if os.path.exists(os.path.join(search_path, lib)):
                        print(f"    ✓ {lib} found at {os.path.join(search_path, lib)}")
                        found = True
                        break
                if not found:
                    print(f"    ✗ {lib} not found")
                    
except Exception as e:
    print(f"  ✗ Error during CUDA check: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
