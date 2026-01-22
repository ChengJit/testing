"""
Main Inventory Door Monitor Application.
Ties together all components for a complete monitoring solution.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from queue import Queue, Empty
from datetime import datetime

import cv2
import numpy as np

from .config import Config
from .detectors import PersonDetector, FaceRecognizer, BoxDetector
from .trackers import ByteTracker, TrackedObject
from .core import ZoneManager, EventManager, EventType, PersonStateMachine
from .utils import JetsonOptimizer, VideoCapture

logger = logging.getLogger(__name__)


@dataclass
class BoxInfo:
    """Information about a detected box."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str
    method: str  # detection method used


@dataclass
class ProcessingResult:
    """Result from AI processing pipeline."""
    frame_id: int
    persons: List[Tuple[Tuple[int, int, int, int], float]]  # (bbox, conf)
    faces: Dict[int, Tuple[str, float]]  # track_id -> (name, confidence)
    boxes: Dict[int, int]  # track_id -> box_count
    box_detections: Dict[int, List[BoxInfo]]  # track_id -> list of box info
    all_boxes: List[BoxInfo]  # all detected boxes for visualization
    processing_time_ms: float


class AIWorker:
    """
    Background AI processing worker.
    Runs detection and recognition in a separate thread.
    """

    def __init__(
        self,
        person_detector: PersonDetector,
        face_recognizer: FaceRecognizer,
        box_detector: BoxDetector,
        process_fps: int = 15,
    ):
        self.person_detector = person_detector
        self.face_recognizer = face_recognizer
        self.box_detector = box_detector
        self.process_interval = 1.0 / process_fps

        self._input_queue: Queue = Queue(maxsize=2)
        self._output_queue: Queue = Queue(maxsize=2)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0

    def start(self):
        """Start the AI worker thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.info("AI worker started")

    def stop(self):
        """Stop the AI worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("AI worker stopped")

    def submit(self, frame: np.ndarray, tracks: Dict[int, TrackedObject]) -> bool:
        """
        Submit frame for processing.

        Returns True if submitted, False if queue full.
        """
        try:
            # Clear old items
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except Empty:
                    break

            self._frame_count += 1
            self._input_queue.put_nowait((frame.copy(), tracks.copy(), self._frame_count))
            return True
        except Exception:
            return False

    def get_result(self, timeout: float = 0.1) -> Optional[ProcessingResult]:
        """Get processing result if available."""
        try:
            return self._output_queue.get(timeout=timeout)
        except Empty:
            return None

    def _worker_loop(self):
        """Main worker loop."""
        while self._running:
            try:
                frame, tracks, frame_id = self._input_queue.get(timeout=0.1)
            except Empty:
                continue

            start_time = time.time()

            try:
                result = self._process_frame(frame, tracks, frame_id)

                # Put result, dropping old ones
                while not self._output_queue.empty():
                    try:
                        self._output_queue.get_nowait()
                    except Empty:
                        break

                self._output_queue.put_nowait(result)

            except Exception as e:
                logger.error(f"AI processing error: {e}")

            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = self.process_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_frame(
        self,
        frame: np.ndarray,
        tracks: Dict[int, TrackedObject],
        frame_id: int
    ) -> ProcessingResult:
        """Process single frame through AI pipeline."""
        start_time = time.time()

        # 1. Detect persons
        person_detections = self.person_detector.detect(frame)
        persons = [(d.bbox, d.confidence) for d in person_detections]

        # 2. Face recognition for each tracked person
        faces = {}
        for track_id, track in tracks.items():
            if track.identity_locked:
                # Already identified
                faces[track_id] = (track.identity, track.identity_confidence)
                continue

            # Run face recognition
            face_match = self.face_recognizer.recognize(
                frame,
                person_bbox=track.bbox,
                track_id=track_id
            )

            if face_match and face_match.name != "Unknown":
                faces[track_id] = (face_match.name, face_match.confidence)

        # 3. Box detection for each person
        boxes = {}
        box_detections = {}
        all_boxes = []

        for track_id, track in tracks.items():
            carried_boxes = self.box_detector.detect_carried_boxes(frame, track.bbox)
            boxes[track_id] = len(carried_boxes)

            # Store detailed box info
            track_box_info = []
            for box in carried_boxes:
                box_info = BoxInfo(
                    bbox=box.bbox,
                    confidence=box.confidence,
                    class_name=box.class_name,
                    method=box.detection_method
                )
                track_box_info.append(box_info)
                all_boxes.append(box_info)

            box_detections[track_id] = track_box_info

            # Log box detections
            if carried_boxes:
                identity = tracks[track_id].identity or f"Track_{track_id}"
                logger.info(
                    f"[BOX] {identity} carrying {len(carried_boxes)} box(es): "
                    f"{[f'{b.class_name}({b.confidence:.0%})' for b in carried_boxes]}"
                )

        processing_time = (time.time() - start_time) * 1000

        return ProcessingResult(
            frame_id=frame_id,
            persons=persons,
            faces=faces,
            boxes=boxes,
            box_detections=box_detections,
            all_boxes=all_boxes,
            processing_time_ms=processing_time
        )


class InventoryMonitor:
    """
    Main inventory monitoring application.

    Orchestrates video capture, AI processing, tracking,
    and event management.
    """

    def __init__(self, config: Config):
        self.config = config
        self.running = False

        # Initialize Jetson optimizer
        self.optimizer = JetsonOptimizer()
        self.optimizer.optimize_for_inference()

        # Get optimal settings
        optimal = self.optimizer.get_optimal_settings()
        logger.info(f"Using optimal settings: {optimal}")

        # Initialize video capture
        self.video = VideoCapture(
            source=config.camera.source,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            buffer_size=config.camera.buffer_size,
        )

        # Initialize detectors
        self.person_detector = PersonDetector(
            model_path=config.detection.yolo_model,
            imgsz=optimal["imgsz"],
            conf_threshold=config.detection.yolo_conf,
            use_tensorrt=config.jetson.use_tensorrt,
            fp16=config.jetson.fp16_inference,
        )

        self.face_recognizer = FaceRecognizer(
            model_name=config.detection.face_model,
            det_size=config.detection.face_det_size,
            recognition_threshold=config.detection.face_recognition_threshold,
            lock_threshold=config.detection.face_lock_threshold,
            faces_dir=str(config.faces_dir),
        )

        self.box_detector = BoxDetector(
            custom_model_path=config.detection.box_model,
            conf_threshold=config.detection.box_conf,
            min_area=config.detection.box_min_area,
            max_area=config.detection.box_max_area,
            imgsz=optimal["imgsz"],
        )

        # Initialize tracker
        self.tracker = ByteTracker(
            track_thresh=config.tracking.track_thresh,
            track_buffer=config.tracking.track_buffer,
            match_thresh=config.tracking.match_thresh,
        )

        # Initialize managers (will be set after first frame)
        self.zone_manager: Optional[ZoneManager] = None
        self.state_machine: Optional[PersonStateMachine] = None
        self.event_manager = EventManager(
            log_dir=str(config.log_dir),
            enable_csv=True,
            enable_json=True,
        )

        # Initialize AI worker
        self.ai_worker = AIWorker(
            person_detector=self.person_detector,
            face_recognizer=self.face_recognizer,
            box_detector=self.box_detector,
            process_fps=optimal["process_fps"],
        )

        # Stats
        self.frame_count = 0
        self.last_ai_result: Optional[ProcessingResult] = None
        self.start_time = 0.0

        # Display settings
        self.show_zones = config.display.show_zones
        self.show_stats = config.display.show_stats
        self.show_boxes = config.display.show_boxes
        self.display_width = config.display.width
        self.display_height = config.display.height

    def start(self):
        """Start the monitoring application."""
        logger.info("Starting Inventory Monitor...")

        # Start video capture
        if not self.video.start():
            logger.error("Failed to start video capture")
            return False

        # Warmup detectors
        logger.info("Warming up AI models...")
        self.person_detector.warmup()

        # Start AI worker
        self.ai_worker.start()

        self.running = True
        self.start_time = time.time()

        logger.info("Inventory Monitor started successfully")
        return True

    def stop(self):
        """Stop the monitoring application."""
        self.running = False
        self.ai_worker.stop()
        self.video.stop()

        # Generate final report
        self.event_manager.generate_daily_report()

        logger.info("Inventory Monitor stopped")

    def run(self):
        """Main application loop."""
        if not self.start():
            return

        try:
            while self.running:
                # Read frame
                ret, frame = self.video.read(timeout=1.0)

                if not ret or frame is None:
                    continue

                self.frame_count += 1

                # Initialize zone/state managers on first frame
                if self.zone_manager is None:
                    h, w = frame.shape[:2]
                    self._init_managers(h, w)

                # Submit to AI worker
                tracks = self.tracker.get_confirmed_tracks()
                self.ai_worker.submit(frame, tracks)

                # Get AI result
                result = self.ai_worker.get_result(timeout=0.01)
                if result:
                    self.last_ai_result = result
                    self._process_ai_result(result)

                # Update tracker with latest detections
                if self.last_ai_result:
                    self.tracker.update(self.last_ai_result.persons)

                # Draw and display
                if not self.config.headless:
                    display_frame = self._draw_overlay(frame)

                    # Resize to 1080p for display
                    display_frame = cv2.resize(
                        display_frame,
                        (self.display_width, self.display_height),
                        interpolation=cv2.INTER_LINEAR
                    )

                    cv2.imshow("Inventory Monitor", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('z'):
                        self.show_zones = not self.show_zones
                    elif key == ord('s'):
                        self.show_stats = not self.show_stats
                    elif key == ord('b'):
                        self.show_boxes = not self.show_boxes
                    elif key == ord('r'):
                        self._handle_registration()
                    elif key == ord('t'):
                        # Retrain faces from images
                        logger.info("Retraining faces from images...")
                        count = self.face_recognizer.retrain_all()
                        logger.info(f"Retrained {count} faces")

                # Memory cleanup periodically
                if self.frame_count % self.config.jetson.clear_cache_interval == 0:
                    self.optimizer.clear_gpu_memory()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
            cv2.destroyAllWindows()

    def _init_managers(self, height: int, width: int):
        """Initialize zone and state managers."""
        door_y = int(self.config.zone.door_line * height)

        self.zone_manager = ZoneManager(
            frame_height=height,
            frame_width=width,
            door_line=self.config.zone.door_line,
            enter_direction_down=self.config.zone.enter_direction_down,
            direction_frames=self.config.tracking.direction_frames,
        )

        self.state_machine = PersonStateMachine(
            door_y=door_y,
            enter_direction_down=self.config.zone.enter_direction_down,
        )

        logger.info(f"Managers initialized for {width}x{height} frame")

    def _process_ai_result(self, result: ProcessingResult):
        """Process AI result and update state."""
        tracks = self.tracker.get_confirmed_tracks()

        for track_id, track in tracks.items():
            # Update identity
            if track_id in result.faces:
                name, conf = result.faces[track_id]
                track.set_identity(name, conf)

            # Update box count
            if track_id in result.boxes:
                track.update_box_count(result.boxes[track_id])

            # Update zone/state
            if self.zone_manager and self.state_machine:
                event = self.state_machine.update(
                    track_id=track_id,
                    center=track.center,
                    box_count=track.box_count,
                    identity=track.identity,
                    identity_confidence=track.identity_confidence,
                )

                # Record events
                if event == "entered":
                    ctx = self.state_machine.get_context(track_id)
                    self.event_manager.record_entry(
                        track_id=track_id,
                        identity=track.identity,
                        identity_confidence=track.identity_confidence,
                        box_count=ctx.entry_box_count if ctx else track.box_count,
                    )
                elif event == "exited":
                    ctx = self.state_machine.get_context(track_id)
                    self.event_manager.record_exit(
                        track_id=track_id,
                        identity=track.identity,
                        identity_confidence=track.identity_confidence,
                        box_count=ctx.exit_box_count if ctx else track.box_count,
                        entry_box_count=ctx.entry_box_count if ctx else 0,
                    )

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw visualization overlay on frame."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw zones
        if self.show_zones and self.zone_manager:
            zone_info = self.zone_manager.get_zone_info()

            # Door line
            cv2.line(display, (0, zone_info["door_line"]), (w, zone_info["door_line"]),
                     (0, 255, 255), 2)
            cv2.putText(display, "DOOR LINE", (10, zone_info["door_line"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Entry zone
            cv2.rectangle(display,
                          (0, zone_info["entry_zone"][0]),
                          (w, zone_info["entry_zone"][1]),
                          (0, 255, 0), 1)

            # Exit zone
            cv2.rectangle(display,
                          (0, zone_info["exit_zone"][0]),
                          (w, zone_info["exit_zone"][1]),
                          (0, 0, 255), 1)

        # Draw detected boxes
        if self.show_boxes and self.last_ai_result:
            for box_info in self.last_ai_result.all_boxes:
                bx1, by1, bx2, by2 = box_info.bbox

                # Cyan color for boxes
                box_color = (255, 255, 0)  # Cyan

                # Draw box bounding box
                cv2.rectangle(display, (bx1, by1), (bx2, by2), box_color, 2)

                # Draw box label
                box_label = f"{box_info.class_name} {box_info.confidence:.0%}"
                cv2.putText(display, box_label, (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Draw detection method indicator
                method_short = box_info.method[:3].upper()
                cv2.putText(display, method_short, (bx2 - 30, by2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

        # Draw tracked persons
        tracks = self.tracker.get_confirmed_tracks()

        for track_id, track in tracks.items():
            x1, y1, x2, y2 = track.bbox

            # Color based on identity status
            if track.identity_locked:
                color = (0, 255, 0)  # Green - identified
            elif track.identity:
                color = (0, 255, 255)  # Yellow - tentative
            else:
                color = (0, 165, 255)  # Orange - unknown

            # Bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # Label
            name = track.identity or f"ID:{track_id}"
            conf_str = f"{track.identity_confidence:.0%}" if track.identity else ""
            box_str = f" [{track.box_count} box]" if track.box_count > 0 else ""

            label = f"{name} {conf_str}{box_str}".strip()

            # Draw label background for readability
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(display, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Direction arrow
            direction = track.get_direction()
            if direction:
                cx, cy = track.center
                arrow_color = (0, 255, 0) if direction == "entering" else (0, 0, 255)
                dy = 40 if direction == "entering" else -40
                cv2.arrowedLine(display, (cx, cy), (cx, cy + dy), arrow_color, 3)

                # Direction label
                dir_label = "ENTERING" if direction == "entering" else "EXITING"
                cv2.putText(display, dir_label, (cx - 40, cy + dy + (20 if dy > 0 else -10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, arrow_color, 2)

        # Draw stats panel
        if self.show_stats:
            stats = self.event_manager.get_statistics()
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0

            # Semi-transparent background for stats
            overlay = display.copy()
            cv2.rectangle(overlay, (5, 5), (250, 160), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

            stats_text = [
                f"FPS: {fps:.1f}",
                f"AI: {self.last_ai_result.processing_time_ms:.0f}ms" if self.last_ai_result else "AI: --",
                f"Entries: {stats['total_entries']} | Exits: {stats['total_exits']}",
                f"Boxes In: {stats['boxes_brought_in']}",
                f"Boxes Out: {stats['boxes_taken_out']}",
                f"People Inside: {stats['persons_currently_inside']}",
            ]

            y = 25
            for text in stats_text:
                cv2.putText(display, text, (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                y += 22

        # Instructions bar at bottom
        instructions = "Q:Quit | Z:Zones | S:Stats | B:Boxes | R:Register | T:Retrain"
        cv2.rectangle(display, (0, h - 25), (w, h), (50, 50, 50), -1)
        cv2.putText(display, instructions, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return display

    def _handle_registration(self):
        """Handle face registration for unknown persons."""
        tracks = self.tracker.get_confirmed_tracks()

        for track_id, track in tracks.items():
            if track.identity_locked:
                continue

            if self.face_recognizer.get_registration_ready(track_id):
                # Get name from user
                name = input(f"Enter name for Track {track_id} (or skip): ").strip()

                if name:
                    success = self.face_recognizer.register_face(track_id, name)
                    if success:
                        track.set_identity(name, 1.0, lock=True)
                        logger.info(f"Registered: {name}")


def main():
    """Entry point for the inventory monitor."""
    import argparse

    parser = argparse.ArgumentParser(description="Inventory Door Monitor")
    parser.add_argument("--config", "-c", default="config.json",
                        help="Path to config file")
    parser.add_argument("--source", "-s", help="Video source (overrides config)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Load config
    config = Config.load(args.config)

    if args.source:
        config.camera.source = args.source
    if args.headless:
        config.headless = True
    if args.debug:
        config.debug = True

    # Save config for future runs
    config.save(args.config)

    # Run monitor
    monitor = InventoryMonitor(config)
    monitor.run()


if __name__ == "__main__":
    main()
