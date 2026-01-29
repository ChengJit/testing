"""
Configuration management for the inventory monitor system.
Optimized defaults for Jetson Orin Nano (8GB VRAM).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional


@dataclass
class DetectionConfig:
    """Detection model configuration."""
    # YOLO settings - optimized for Jetson
    yolo_model: str = "yolo11n.pt"  # Nano model for speed
    yolo_imgsz: int = 480  # Balance between accuracy and speed
    yolo_conf: float = 0.5  # Confidence threshold
    yolo_iou: float = 0.45  # NMS IoU threshold

    # Box detection
    box_model: Optional[str] = None  # Custom trained box model
    box_conf: float = 0.4
    box_min_area: int = 1500  # Minimum box area in pixels
    box_max_area: int = 80000  # Maximum box area

    # Face recognition
    face_model: str = "buffalo_s"  # Smaller model for Jetson
    face_det_size: Tuple[int, int] = (320, 320)
    face_recognition_threshold: float = 0.45
    face_lock_threshold: float = 0.75  # Lock identity at this confidence


@dataclass
class TrackingConfig:
    """Tracking configuration."""
    # ByteTrack parameters
    track_thresh: float = 0.5  # Detection threshold for tracking
    track_buffer: int = 30  # Frames to keep lost tracks
    match_thresh: float = 0.8  # IoU threshold for matching

    # Zone configuration (percentage of frame height)
    entry_zone_start: float = 0.0  # Top of frame
    entry_zone_end: float = 0.25  # 25% from top
    exit_zone_start: float = 0.75  # 75% from top
    exit_zone_end: float = 1.0  # Bottom of frame

    # Direction confirmation
    direction_frames: int = 5  # Frames to confirm direction
    min_movement: int = 20  # Minimum pixels to detect movement


@dataclass
class JetsonConfig:
    """Jetson Orin Nano specific optimizations."""
    use_tensorrt: bool = True  # Use TensorRT acceleration
    use_dla: bool = False  # Use DLA cores (if available)
    fp16_inference: bool = True  # Half precision for speed
    max_batch_size: int = 1  # Single image batching

    # Memory management
    gpu_memory_fraction: float = 0.7  # Reserve 70% GPU memory
    clear_cache_interval: int = 50  # Clear CUDA cache every N frames

    # Processing
    process_fps: int = 15  # Target AI processing FPS
    display_fps: int = 30  # Display refresh rate
    queue_size: int = 2  # Frame queue size (smaller = less latency)


@dataclass
class CameraConfig:
    """Camera/video source configuration."""
    source: str = "rtsp://fasspay:fasspay2025@192.168.122.127:554/stream1"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    buffer_size: int = 1  # Minimize latency
    reconnect_delay: float = 5.0  # Seconds before reconnect attempt


@dataclass
class DisplayConfig:
    """Display configuration."""
    width: int = 1920
    height: int = 1080
    show_zones: bool = True
    show_stats: bool = True
    show_boxes: bool = True


@dataclass
class ZoneConfig:
    """Door/monitoring zone configuration."""
    # Region of interest (normalized 0-1)
    roi_x1: float = 0.0
    roi_y1: float = 0.0
    roi_x2: float = 1.0
    roi_y2: float = 1.0

    # Door line position (normalized, where 0=top, 1=bottom)
    door_line: float = 0.5

    # Direction: True = entering when moving down, False = entering when moving up
    enter_direction_down: bool = True


@dataclass
class TrainingConfig:
    """Training data collection configuration."""
    enabled: bool = False
    output_dir: str = "training_queue"
    capture_interval: int = 30  # frames between captures per track
    max_samples: int = 5000
    save_full_frame: bool = False
    negative_ratio: float = 0.5  # target ratio of negative samples


@dataclass
class ReviewConfig:
    """Review queue configuration for low-confidence detections."""
    threshold: float = 0.6      # face confidence below this -> review queue
    max_queue_size: int = 50


@dataclass
class Config:
    """Main configuration container."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    jetson: JetsonConfig = field(default_factory=JetsonConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    faces_dir: Path = field(default_factory=lambda: Path("known_faces"))

    # Runtime
    headless: bool = False
    debug: bool = False

    def __post_init__(self):
        """Ensure paths exist."""
        for path in [self.data_dir, self.log_dir, self.models_dir, self.faces_dir]:
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path: str = "config.json") -> "Config":
        """Load configuration from JSON file."""
        config = cls()
        config_path = Path(path)

        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)

            # Update nested configs
            if "detection" in data:
                for k, v in data["detection"].items():
                    if hasattr(config.detection, k):
                        setattr(config.detection, k, v)

            if "tracking" in data:
                for k, v in data["tracking"].items():
                    if hasattr(config.tracking, k):
                        setattr(config.tracking, k, v)

            if "jetson" in data:
                for k, v in data["jetson"].items():
                    if hasattr(config.jetson, k):
                        setattr(config.jetson, k, v)

            if "camera" in data:
                for k, v in data["camera"].items():
                    if hasattr(config.camera, k):
                        setattr(config.camera, k, v)

            if "zone" in data:
                for k, v in data["zone"].items():
                    if hasattr(config.zone, k):
                        setattr(config.zone, k, v)

            if "training" in data:
                for k, v in data["training"].items():
                    if hasattr(config.training, k):
                        setattr(config.training, k, v)

            if "review" in data:
                for k, v in data["review"].items():
                    if hasattr(config.review, k):
                        setattr(config.review, k, v)

        return config

    def save(self, path: str = "config.json"):
        """Save configuration to JSON file."""
        data = {
            "detection": {
                "yolo_model": self.detection.yolo_model,
                "yolo_imgsz": self.detection.yolo_imgsz,
                "yolo_conf": self.detection.yolo_conf,
                "box_model": self.detection.box_model,
                "box_conf": self.detection.box_conf,
                "face_model": self.detection.face_model,
                "face_det_size": list(self.detection.face_det_size),
                "face_recognition_threshold": self.detection.face_recognition_threshold,
            },
            "tracking": {
                "track_thresh": self.tracking.track_thresh,
                "track_buffer": self.tracking.track_buffer,
                "entry_zone_end": self.tracking.entry_zone_end,
                "exit_zone_start": self.tracking.exit_zone_start,
                "direction_frames": self.tracking.direction_frames,
            },
            "jetson": {
                "use_tensorrt": self.jetson.use_tensorrt,
                "fp16_inference": self.jetson.fp16_inference,
                "process_fps": self.jetson.process_fps,
            },
            "camera": {
                "source": self.camera.source,
                "width": self.camera.width,
                "height": self.camera.height,
            },
            "zone": {
                "roi_x1": self.zone.roi_x1,
                "roi_y1": self.zone.roi_y1,
                "roi_x2": self.zone.roi_x2,
                "roi_y2": self.zone.roi_y2,
                "door_line": self.zone.door_line,
                "enter_direction_down": self.zone.enter_direction_down,
            },
            "review": {
                "threshold": self.review.threshold,
                "max_queue_size": self.review.max_queue_size,
            }
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# Default configuration instance
default_config = Config()
