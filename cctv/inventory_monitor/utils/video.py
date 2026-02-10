"""
Video capture utilities optimized for RTSP streams.
"""

import logging
import time
import threading
from typing import Optional, Tuple
from queue import Queue, Empty
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Frame with metadata."""
    frame: np.ndarray
    timestamp: float
    frame_id: int


class FrameBuffer:
    """
    Thread-safe frame buffer for low-latency video processing.

    Keeps only the most recent frame to minimize latency.
    """

    def __init__(self, maxsize: int = 2):
        self._queue: Queue = Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._latest_frame: Optional[FrameData] = None

    def put(self, frame_data: FrameData):
        """Add frame to buffer, dropping old frames if full."""
        with self._lock:
            self._latest_frame = frame_data

            # Clear queue and add new frame
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Empty:
                    break

            try:
                self._queue.put_nowait(frame_data)
            except Exception:
                pass

    def get(self, timeout: float = 1.0) -> Optional[FrameData]:
        """Get latest frame from buffer."""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            with self._lock:
                return self._latest_frame

    def get_latest(self) -> Optional[FrameData]:
        """Get the most recent frame without waiting."""
        with self._lock:
            return self._latest_frame

    def clear(self):
        """Clear all frames from buffer."""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
            self._latest_frame = None


class VideoCapture:
    """
    Robust video capture with automatic reconnection.

    Features:
    - Threaded frame capture for low latency
    - Automatic reconnection on failure
    - RTSP-optimized settings
    - Frame dropping to maintain real-time
    """

    def __init__(
        self,
        source: str,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        buffer_size: int = 1,
        reconnect_delay: float = 5.0,
        use_gstreamer: bool = True,
        capture_width: int = 0,
        capture_height: int = 0,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.use_gstreamer = use_gstreamer
        # Target capture resolution (0 = no downscale, use native)
        self.capture_width = capture_width
        self.capture_height = capture_height

        self._cap: Optional[cv2.VideoCapture] = None
        self._buffer = FrameBuffer(maxsize=2)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._start_time = 0.0
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Start video capture."""
        if self._running:
            return True

        if not self._connect():
            return False

        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info(f"Video capture started: {self.source}")
        return True

    def stop(self):
        """Stop video capture."""
        self._running = False

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap:
            self._cap.release()
            self._cap = None

        logger.info("Video capture stopped")

    def _connect(self) -> bool:
        """Connect to video source."""
        if self._cap:
            self._cap.release()

        # Build pipeline
        pipeline = self._build_pipeline()

        try:
            if pipeline and self.use_gstreamer:
                self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                self._cap = cv2.VideoCapture(self.source)

            if not self._cap.isOpened():
                # Fallback to direct connection
                self._cap = cv2.VideoCapture(self.source)

            if self._cap.isOpened():
                # Set properties
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

                # Get actual resolution
                actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

                logger.info(
                    f"Connected: {actual_width}x{actual_height} @ {actual_fps:.1f}fps"
                )
                return True
            else:
                logger.error(f"Failed to open video source: {self.source}")
                return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def _build_pipeline(self) -> Optional[str]:
        """Build GStreamer pipeline for RTSP streams."""
        if not self.source.startswith("rtsp://"):
            return None

        # Optional hardware-accelerated downscale via nvvidconv
        if self.capture_width > 0 and self.capture_height > 0:
            scale_cap = (
                f"video/x-raw(memory:NVMM),width={self.capture_width},"
                f"height={self.capture_height} ! "
            )
        else:
            scale_cap = ""

        # Extract base URL (remove query params for GStreamer)
        rtsp_url = self.source.split("?")[0]

        # Check if TCP transport is requested via URL param
        use_tcp = "rtsp_transport=tcp" in self.source
        protocols = "protocols=tcp" if use_tcp else ""

        # GStreamer pipeline optimized for Jetson/RTSP
        pipeline = (
            f"rtspsrc location={rtsp_url} {protocols} latency=0 ! "
            f"rtph264depay ! h264parse ! "
            f"nvv4l2decoder ! "
            f"nvvidconv ! "
            f"{scale_cap}"
            f"video/x-raw,format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=1 max-buffers=1"
        )

        return pipeline

    def _capture_loop(self):
        """Main capture loop running in background thread."""
        consecutive_failures = 0
        max_failures = 10

        while self._running:
            if not self._cap or not self._cap.isOpened():
                logger.warning("Connection lost, attempting reconnect...")
                time.sleep(self.reconnect_delay)
                if not self._connect():
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.error("Max reconnection attempts reached")
                        break
                    continue
                consecutive_failures = 0

            try:
                ret, frame = self._cap.read()

                if ret and frame is not None:
                    consecutive_failures = 0
                    self._frame_count += 1

                    # Software downscale if capture size is set and
                    # the frame is larger (GStreamer may already have
                    # handled this; this covers the fallback path).
                    cw, ch = self.capture_width, self.capture_height
                    if cw > 0 and ch > 0:
                        fh, fw = frame.shape[:2]
                        if fw > cw or fh > ch:
                            frame = cv2.resize(
                                frame, (cw, ch),
                                interpolation=cv2.INTER_LINEAR)

                    frame_data = FrameData(
                        frame=frame,
                        timestamp=time.time(),
                        frame_id=self._frame_count
                    )
                    self._buffer.put(frame_data)

                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.warning("Too many read failures, reconnecting...")
                        self._cap.release()
                        self._cap = None
                        consecutive_failures = 0

            except Exception as e:
                logger.error(f"Capture error: {e}")
                consecutive_failures += 1

            # Small sleep to prevent CPU spinning
            time.sleep(0.001)

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest frame.

        Args:
            timeout: Maximum time to wait for frame

        Returns:
            (success, frame) tuple
        """
        frame_data = self._buffer.get(timeout=timeout)

        if frame_data is None:
            return False, None

        return True, frame_data.frame

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the most recent frame without waiting."""
        frame_data = self._buffer.get_latest()

        if frame_data is None:
            return False, None

        return True, frame_data.frame

    def get_fps(self) -> float:
        """Get actual capture FPS."""
        if self._start_time == 0:
            return 0.0

        elapsed = time.time() - self._start_time
        if elapsed > 0:
            return self._frame_count / elapsed
        return 0.0

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running and self._cap is not None and self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count
