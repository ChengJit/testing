"""
Jetson Orin Nano specific optimizations and utilities.
"""

import logging
import os
import subprocess
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def get_gpu_info() -> Dict:
    """Get GPU/Jetson information."""
    info = {
        "platform": "unknown",
        "gpu_available": False,
        "cuda_available": False,
        "tensorrt_available": False,
        "gpu_memory_mb": 0,
        "gpu_memory_used_mb": 0,
    }

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()

        if info["cuda_available"]:
            info["gpu_available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)

            # Check if this is a Jetson
            if "orin" in info["device_name"].lower() or "tegra" in info["device_name"].lower():
                info["platform"] = "jetson"
            else:
                info["platform"] = "cuda"

    except ImportError:
        pass

    # Check TensorRT availability
    try:
        import tensorrt
        info["tensorrt_available"] = True
        info["tensorrt_version"] = tensorrt.__version__
    except ImportError:
        pass

    # Try to get Jetson-specific info
    try:
        if os.path.exists("/etc/nv_tegra_release"):
            with open("/etc/nv_tegra_release") as f:
                info["tegra_release"] = f.read().strip()
            info["platform"] = "jetson"
    except Exception:
        pass

    return info


class JetsonOptimizer:
    """
    Jetson-specific optimizations for deep learning inference.

    Features:
    - Power mode management
    - GPU clock optimization
    - Memory management
    - TensorRT integration helpers
    """

    # Jetson power modes
    POWER_MODES = {
        "maxn": 0,      # Maximum performance (all cores, max clocks)
        "15w": 1,       # 15W power budget
        "10w": 2,       # 10W power budget
        "7w": 3,        # 7W power budget (default for Orin Nano)
    }

    def __init__(self, power_mode: str = "15w"):
        self.gpu_info = get_gpu_info()
        self.is_jetson = self.gpu_info["platform"] == "jetson"

        if self.is_jetson:
            logger.info(f"Running on Jetson: {self.gpu_info.get('device_name', 'Unknown')}")
            self._set_power_mode(power_mode)
        else:
            logger.info(f"Running on: {self.gpu_info.get('platform', 'CPU')}")

    def _set_power_mode(self, mode: str):
        """Set Jetson power mode."""
        if not self.is_jetson:
            return

        mode_id = self.POWER_MODES.get(mode, 1)

        try:
            # nvpmodel requires sudo on Jetson
            result = subprocess.run(
                ["sudo", "nvpmodel", "-m", str(mode_id)],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Set Jetson power mode to: {mode}")
            else:
                logger.warning(f"Could not set power mode: {result.stderr.decode()}")
        except Exception as e:
            logger.warning(f"Power mode setting skipped: {e}")

    def optimize_for_inference(self):
        """Apply optimizations for inference workload."""
        if not self.gpu_info["cuda_available"]:
            return

        try:
            import torch

            # Enable cudnn benchmarking for consistent input sizes
            torch.backends.cudnn.benchmark = True

            # Use TF32 on Ampere+ GPUs for speed
            if hasattr(torch.backends, 'cuda'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            logger.info("Applied PyTorch optimizations for inference")

        except Exception as e:
            logger.warning(f"Could not apply PyTorch optimizations: {e}")

    def clear_gpu_memory(self):
        """Clear GPU memory cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    def get_memory_usage(self) -> Tuple[int, int]:
        """Get GPU memory usage (used_mb, total_mb)."""
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() // (1024 * 1024)
                total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                return (used, total)
        except Exception:
            pass
        return (0, 0)

    def export_to_tensorrt(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        imgsz: int = 480,
        fp16: bool = True,
        workspace_gb: int = 4,
    ) -> Optional[str]:
        """
        Export YOLO model to TensorRT engine.

        Args:
            model_path: Path to YOLO .pt model
            output_path: Output .engine path (auto-generated if None)
            imgsz: Input image size
            fp16: Use FP16 precision
            workspace_gb: TensorRT workspace size in GB

        Returns:
            Path to exported engine, or None if failed
        """
        if not self.gpu_info["tensorrt_available"]:
            logger.warning("TensorRT not available")
            return None

        try:
            from ultralytics import YOLO
            from pathlib import Path

            model = YOLO(model_path)

            # Export to TensorRT
            engine_path = model.export(
                format="engine",
                imgsz=imgsz,
                half=fp16,
                simplify=True,
                workspace=workspace_gb,
            )

            logger.info(f"Exported TensorRT engine: {engine_path}")
            return str(engine_path)

        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            return None

    def get_optimal_settings(self) -> Dict:
        """
        Get optimal settings for current hardware.

        Returns recommended settings for:
        - Image size
        - Batch size
        - FP16 usage
        - Thread count
        """
        settings = {
            "imgsz": 480,
            "batch_size": 1,
            "fp16": True,
            "num_threads": 4,
            "process_fps": 15,
        }

        if self.is_jetson:
            # Jetson Orin Nano optimized settings
            used_mb, total_mb = self.get_memory_usage()
            available_mb = total_mb - used_mb

            if available_mb > 6000:
                settings["imgsz"] = 640
                settings["process_fps"] = 20
            elif available_mb > 4000:
                settings["imgsz"] = 480
                settings["process_fps"] = 15
            else:
                settings["imgsz"] = 416
                settings["process_fps"] = 12

            settings["num_threads"] = 6  # Orin Nano has 6 CPU cores

        elif self.gpu_info["cuda_available"]:
            # Desktop GPU - can handle more
            settings["imgsz"] = 640
            settings["process_fps"] = 30

        else:
            # CPU only - conservative settings
            settings["imgsz"] = 416
            settings["fp16"] = False
            settings["process_fps"] = 10

        return settings
