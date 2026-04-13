"""
Multi-Camera Viewer - Display multiple camera feeds in a grid.
"""

import cv2
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Tuple
from queue import Queue, Empty
from dataclasses import dataclass

from .config import CameraSource, RackInventoryConfig


@dataclass
class CameraFrame:
    """Frame data from a camera."""
    name: str
    frame: np.ndarray
    timestamp: float
    fps: float = 0.0


class CameraStream:
    """Handles a single camera RTSP stream."""

    def __init__(self, source: CameraSource):
        self.source = source
        self.name = source.name
        self.url = source.rtsp_url

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self._frame_count = 0
        self._start_time = 0.0
        self._fps = 0.0
        self._connected = False
        self._last_frame_time = 0.0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def fps(self) -> float:
        return self._fps

    def start(self) -> bool:
        """Start the camera stream."""
        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop the camera stream."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _capture_loop(self):
        """Main capture loop."""
        reconnect_delay = 2.0

        while self._running:
            # Try to connect
            try:
                # Use FFMPEG with low-latency settings
                self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

                # Minimize buffer for low latency
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS

                if not self._cap.isOpened():
                    print(f"[{self.name}] Failed to connect to {self.source.ip}")
                    self._connected = False
                    time.sleep(reconnect_delay)
                    continue

                self._connected = True
                self._start_time = time.time()
                self._frame_count = 0
                print(f"[{self.name}] Connected to {self.source.ip}")

                # Read frames
                while self._running and self._cap.isOpened():
                    ret, frame = self._cap.read()

                    if not ret:
                        print(f"[{self.name}] Lost connection")
                        self._connected = False
                        break

                    # Skip frames to reduce latency - grab latest
                    self._cap.grab()  # Discard buffered frame

                    with self._lock:
                        self._frame = frame
                        self._last_frame_time = time.time()

                    self._frame_count += 1
                    elapsed = time.time() - self._start_time
                    if elapsed > 0:
                        self._fps = self._frame_count / elapsed

                    # Reset counter periodically
                    if self._frame_count > 300:
                        self._frame_count = 0
                        self._start_time = time.time()

                    # Small delay to prevent CPU overload
                    time.sleep(0.01)

            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                self._connected = False

            finally:
                if self._cap:
                    self._cap.release()
                    self._cap = None

            if self._running:
                time.sleep(reconnect_delay)


class MultiCameraViewer:
    """Display multiple camera feeds in a grid layout."""

    def __init__(self, config: RackInventoryConfig):
        self.config = config
        self.streams: Dict[str, CameraStream] = {}
        self._running = False

    def add_camera(self, source: CameraSource):
        """Add a camera to the viewer."""
        stream = CameraStream(source)
        self.streams[source.name] = stream

    def start(self):
        """Start all camera streams."""
        print(f"Starting {len(self.config.cameras)} camera(s)...")

        for cam in self.config.cameras:
            if cam.enabled:
                self.add_camera(cam)
                self.streams[cam.name].start()

        self._running = True

    def stop(self):
        """Stop all camera streams."""
        self._running = False
        for stream in self.streams.values():
            stream.stop()
        print("All cameras stopped")

    def create_grid(self) -> Optional[np.ndarray]:
        """Create a grid view of all camera feeds."""
        frames = []

        for name, stream in self.streams.items():
            frame = stream.get_frame()

            if frame is None:
                # Create placeholder for disconnected camera
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"{name}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                cv2.putText(frame, "Connecting...", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
            else:
                # Add camera name and status overlay
                frame = frame.copy()
                status_color = (0, 255, 0) if stream.connected else (0, 0, 255)

                # Semi-transparent header
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                # Camera name and FPS
                cv2.putText(frame, f"{name}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"{stream.fps:.1f} FPS", (frame.shape[1] - 100, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                # Connection indicator
                cv2.circle(frame, (frame.shape[1] - 120, 20), 8, status_color, -1)

            frames.append((name, frame))

        if not frames:
            return None

        # Create grid layout
        cols = self.config.grid_columns
        rows = (len(frames) + cols - 1) // cols

        # Calculate cell size
        cell_w = self.config.window_width // cols
        cell_h = self.config.window_height // rows

        # Create output grid
        grid = np.zeros((self.config.window_height, self.config.window_width, 3), dtype=np.uint8)

        for i, (name, frame) in enumerate(frames):
            row = i // cols
            col = i % cols

            # Resize frame to fit cell
            resized = cv2.resize(frame, (cell_w, cell_h))

            # Place in grid
            y1 = row * cell_h
            y2 = y1 + cell_h
            x1 = col * cell_w
            x2 = x1 + cell_w

            grid[y1:y2, x1:x2] = resized

        return grid

    def run(self):
        """Run the viewer with display window."""
        self.start()

        window_name = "Rack Inventory - Multi Camera View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.config.window_width, self.config.window_height)

        print("\nControls:")
        print("  Q - Quit")
        print("  S - Screenshot")
        print("  R - Reconnect all cameras")
        print("-" * 40)

        try:
            while self._running:
                grid = self.create_grid()

                if grid is not None:
                    # Add instructions bar
                    cv2.rectangle(grid, (0, grid.shape[0] - 30),
                                 (grid.shape[1], grid.shape[0]), (40, 40, 40), -1)
                    cv2.putText(grid, "Q:Quit | S:Screenshot | R:Reconnect",
                               (10, grid.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    cv2.imshow(window_name, grid)

                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Screenshot
                    filename = f"rack_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, grid)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('r'):
                    # Reconnect all
                    print("Reconnecting all cameras...")
                    self.stop()
                    time.sleep(1)
                    self.start()

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.stop()
            cv2.destroyAllWindows()


def run_viewer(config: Optional[RackInventoryConfig] = None):
    """Run the multi-camera viewer."""
    if config is None:
        config = RackInventoryConfig.load("rack_config.json")

    if not config.cameras:
        print("No cameras configured!")
        print("Run the scanner first or edit rack_config.json")
        return

    viewer = MultiCameraViewer(config)
    viewer.run()
