"""
Optimized Real-time Multi-Camera Viewer
Uses multiprocessing to bypass Python GIL for better performance.
"""

import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import ctypes

from .config import CameraSource, RackInventoryConfig


def camera_worker(
    name: str,
    url: str,
    frame_queue: mp.Queue,
    running: mp.Value,
    connected: mp.Value,
):
    """
    Camera capture worker - runs in separate process.
    Captures frames and puts latest into queue.
    """
    cap = None
    reconnect_delay = 2.0

    while running.value:
        try:
            # Connect to camera
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                connected.value = 0
                time.sleep(reconnect_delay)
                continue

            connected.value = 1
            frame_count = 0
            start_time = time.time()

            while running.value and cap.isOpened():
                # Grab latest frame (skip buffer)
                cap.grab()
                ret, frame = cap.retrieve()

                if not ret:
                    connected.value = 0
                    break

                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                # Reset counter periodically
                if frame_count > 100:
                    frame_count = 0
                    start_time = time.time()

                # Clear queue and put latest frame
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except:
                        break

                try:
                    # Resize before sending (reduce data transfer)
                    h, w = frame.shape[:2]
                    if w > 640:
                        scale = 640 / w
                        frame = cv2.resize(frame, (640, int(h * scale)))

                    frame_queue.put_nowait((frame, fps, time.time()))
                except:
                    pass

                # Small delay to control frame rate
                time.sleep(0.03)  # ~30 FPS max

        except Exception as e:
            connected.value = 0

        finally:
            if cap:
                cap.release()
                cap = None

        if running.value:
            time.sleep(reconnect_delay)


class RealtimeMultiViewer:
    """
    Real-time multi-camera viewer using multiprocessing.
    Each camera runs in its own process for true parallelism.
    """

    def __init__(self, config: RackInventoryConfig):
        self.config = config
        self.processes: Dict[str, Process] = {}
        self.queues: Dict[str, mp.Queue] = {}
        self.running_flags: Dict[str, mp.Value] = {}
        self.connected_flags: Dict[str, mp.Value] = {}
        self._running = False

    def start(self):
        """Start all camera processes."""
        print(f"Starting {len(self.config.cameras)} camera process(es)...")

        for cam in self.config.cameras:
            if not cam.enabled:
                continue

            # Create shared state
            frame_queue = mp.Queue(maxsize=2)
            running_flag = mp.Value(ctypes.c_bool, True)
            connected_flag = mp.Value(ctypes.c_bool, False)

            # Start camera process
            proc = Process(
                target=camera_worker,
                args=(cam.name, cam.rtsp_url, frame_queue, running_flag, connected_flag),
                daemon=True,
            )
            proc.start()

            self.processes[cam.name] = proc
            self.queues[cam.name] = frame_queue
            self.running_flags[cam.name] = running_flag
            self.connected_flags[cam.name] = connected_flag

            print(f"  Started: {cam.name} ({cam.ip})")

        self._running = True

    def stop(self):
        """Stop all camera processes."""
        self._running = False

        # Signal all processes to stop
        for flag in self.running_flags.values():
            flag.value = False

        # Wait for processes to finish
        for name, proc in self.processes.items():
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
            print(f"  Stopped: {name}")

        self.processes.clear()
        self.queues.clear()

    def get_frames(self) -> Dict[str, tuple]:
        """Get latest frames from all cameras."""
        frames = {}

        for name, queue in self.queues.items():
            try:
                frame_data = queue.get_nowait()
                frames[name] = frame_data
            except:
                frames[name] = None

        return frames

    def create_grid(self, frames: Dict[str, tuple]) -> np.ndarray:
        """Create grid display from frames."""
        display_frames = []

        for cam in self.config.cameras:
            name = cam.name
            frame_data = frames.get(name)
            connected = self.connected_flags.get(name, mp.Value(ctypes.c_bool, False)).value

            if frame_data is not None:
                frame, fps, timestamp = frame_data
                frame = frame.copy()

                # Add overlay
                color = (0, 255, 0) if connected else (0, 165, 255)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(frame, f"{name}", (10, 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"{fps:.0f}fps", (frame.shape[1] - 60, 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.circle(frame, (frame.shape[1] - 75, 18), 5, color, -1)

            else:
                # Placeholder
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                status = "Connecting..." if connected else "Disconnected"
                cv2.putText(frame, name, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                cv2.putText(frame, status, (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

            display_frames.append(frame)

        if not display_frames:
            return np.zeros((360, 640, 3), dtype=np.uint8)

        # Create grid
        cols = self.config.grid_columns
        rows = (len(display_frames) + cols - 1) // cols

        # Ensure all frames same size
        target_h, target_w = 360, 640
        resized = []
        for f in display_frames:
            if f.shape[:2] != (target_h, target_w):
                f = cv2.resize(f, (target_w, target_h))
            resized.append(f)

        # Pad with black frames if needed
        while len(resized) < rows * cols:
            resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

        # Build grid
        grid_rows = []
        for r in range(rows):
            row_frames = resized[r * cols:(r + 1) * cols]
            grid_rows.append(np.hstack(row_frames))

        grid = np.vstack(grid_rows)
        return grid

    def run(self):
        """Run the viewer."""
        self.start()

        window_name = "Rack Inventory - Real-time View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\nControls: Q=Quit, S=Screenshot")
        print("-" * 40)

        last_frames = {}
        fps_counter = 0
        fps_start = time.time()
        display_fps = 0

        try:
            while self._running:
                # Get frames from all cameras
                new_frames = self.get_frames()

                # Update with new frames (keep old if no new)
                for name, data in new_frames.items():
                    if data is not None:
                        last_frames[name] = data

                # Create grid
                grid = self.create_grid(last_frames)

                # Add FPS counter
                fps_counter += 1
                if fps_counter % 30 == 0:
                    display_fps = fps_counter / (time.time() - fps_start)
                    fps_counter = 0
                    fps_start = time.time()

                # Add info bar
                h = grid.shape[0]
                cv2.rectangle(grid, (0, h - 25), (grid.shape[1], h), (40, 40, 40), -1)
                cv2.putText(grid, f"Display: {display_fps:.0f}fps | Q:Quit S:Screenshot",
                           (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow(window_name, grid)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    fname = f"rack_{int(time.time())}.jpg"
                    cv2.imwrite(fname, grid)
                    print(f"Saved: {fname}")

        except KeyboardInterrupt:
            pass

        finally:
            self.stop()
            cv2.destroyAllWindows()


def run_realtime(config: Optional[RackInventoryConfig] = None):
    """Run real-time multi-camera viewer."""
    if config is None:
        config = RackInventoryConfig.load("rack_config.json")

    if not config.cameras:
        print("No cameras configured!")
        return

    viewer = RealtimeMultiViewer(config)
    viewer.run()


if __name__ == "__main__":
    run_realtime()
