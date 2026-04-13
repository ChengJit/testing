"""Configuration for Rack Inventory Scanner."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import json


@dataclass
class CameraSource:
    """Single camera configuration."""
    name: str
    ip: str
    username: str = "fasspay"
    password: str = "fasspay2025"
    port: int = 554
    stream: str = "stream2"  # stream1=HD, stream2=SD (use SD for lower latency)
    enabled: bool = True

    @property
    def rtsp_url(self) -> str:
        """Generate RTSP URL with TCP transport for stability."""
        return f"rtsp://{self.username}:{self.password}@{self.ip}:{self.port}/{self.stream}?rtsp_transport=tcp"


@dataclass
class RackZone:
    """Define a zone on a rack for inventory tracking."""
    name: str
    camera: str  # camera name
    x1: float  # normalized 0-1
    y1: float
    x2: float
    y2: float
    expected_items: int = 0  # expected count for alerts


@dataclass
class RackInventoryConfig:
    """Main configuration for rack inventory system."""

    # Camera sources
    cameras: List[CameraSource] = field(default_factory=list)

    # Rack zones for inventory tracking
    zones: List[RackZone] = field(default_factory=list)

    # Detection settings
    detection_model: str = "yolo11n.pt"
    detection_confidence: float = 0.5
    detection_classes: List[str] = field(default_factory=lambda: ["terminal", "box", "device"])

    # Display settings
    grid_columns: int = 2  # 2 cameras side by side
    window_width: int = 960   # Reduced for better performance
    window_height: int = 540
    show_zones: bool = True
    show_counts: bool = True

    # Scanning settings
    scan_interval: float = 5.0  # seconds between inventory scans
    alert_on_change: bool = True

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("rack_data"))
    log_file: Path = field(default_factory=lambda: Path("rack_data/inventory_log.csv"))

    def __post_init__(self):
        """Ensure directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path: str = "rack_config.json") -> "RackInventoryConfig":
        """Load config from JSON file."""
        config = cls()
        config_path = Path(path)

        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)

            # Load cameras
            if "cameras" in data:
                config.cameras = [
                    CameraSource(**cam) for cam in data["cameras"]
                ]

            # Load zones
            if "zones" in data:
                config.zones = [
                    RackZone(**zone) for zone in data["zones"]
                ]

            # Load other settings
            for key in ["detection_model", "detection_confidence", "grid_columns",
                       "window_width", "window_height", "scan_interval"]:
                if key in data:
                    setattr(config, key, data[key])

        return config

    def save(self, path: str = "rack_config.json"):
        """Save config to JSON file."""
        data = {
            "cameras": [
                {
                    "name": cam.name,
                    "ip": cam.ip,
                    "username": cam.username,
                    "password": cam.password,
                    "port": cam.port,
                    "stream": cam.stream,
                    "enabled": cam.enabled,
                }
                for cam in self.cameras
            ],
            "zones": [
                {
                    "name": zone.name,
                    "camera": zone.camera,
                    "x1": zone.x1,
                    "y1": zone.y1,
                    "x2": zone.x2,
                    "y2": zone.y2,
                    "expected_items": zone.expected_items,
                }
                for zone in self.zones
            ],
            "detection_model": self.detection_model,
            "detection_confidence": self.detection_confidence,
            "grid_columns": self.grid_columns,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "scan_interval": self.scan_interval,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# Default config with placeholder cameras
def create_default_config() -> RackInventoryConfig:
    """Create default config - cameras to be discovered."""
    return RackInventoryConfig(
        cameras=[
            CameraSource(name="Rack-Cam-1", ip="192.168.122.128"),
            CameraSource(name="Rack-Cam-2", ip="192.168.122.129"),
        ],
        zones=[],
    )
