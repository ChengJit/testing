"""
Capture Worker for immediate event logging at door crossings.

Monitors door crossing events, captures full body crops immediately,
logs events with track_id, and queues for async recognition.
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple, TYPE_CHECKING

import numpy as np

from .event_manager import CaptureEvent, EventType

if TYPE_CHECKING:
    from .event_manager import EventManager
    from .recognition_worker import RecognitionQueue
    from ..trackers.bytetrack import TrackedObject

logger = logging.getLogger(__name__)


class CaptureWorker:
    """
    Monitors door crossings and captures body crops for recognition.

    Thread 1 responsibilities:
    - Watch for door crossing events from state machine
    - Capture full body crops at crossing moment
    - Log events immediately (with identity=None if unknown)
    - Queue CaptureEvents for recognition worker
    """

    def __init__(
        self,
        event_manager: "EventManager",
        recognition_queue: "RecognitionQueue",
        door_y: int,
        enter_direction_down: bool = True,
        crossing_threshold: int = 30,
        capture_padding: float = 0.1,
    ):
        """
        Initialize capture worker.

        Args:
            event_manager: For logging immediate events
            recognition_queue: Queue to send captures for recognition
            door_y: Y coordinate of door line
            enter_direction_down: True if moving down = entering
            crossing_threshold: Pixels from door line to trigger capture
            capture_padding: Padding fraction for body crops
        """
        self.event_manager = event_manager
        self.recognition_queue = recognition_queue
        self.door_y = door_y
        self.enter_direction_down = enter_direction_down
        self.crossing_threshold = crossing_threshold
        self.capture_padding = capture_padding

        # Track crossing state per track_id
        self._track_states: Dict[int, str] = {}  # track_id -> "above" | "below" | "crossing"
        self._pending_captures: Dict[int, CaptureEvent] = {}  # track_id -> pending capture

        # Frame reference for capture
        self._current_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # Statistics
        self.stats = {
            "entries_captured": 0,
            "exits_captured": 0,
            "captures_queued": 0,
        }

    def update_frame(self, frame: np.ndarray):
        """
        Update current frame reference for captures.

        Args:
            frame: Current video frame
        """
        with self._frame_lock:
            self._current_frame = frame.copy()
            # Log occasionally to show it's receiving frames
            if hasattr(self, '_frame_count'):
                self._frame_count += 1
            else:
                self._frame_count = 1
            if self._frame_count % 300 == 1:
                logger.debug(f"[CAPTURE_WORKER] Frame #{self._frame_count} received, shape={frame.shape}")

    def process_track(
        self,
        track_id: int,
        track: "TrackedObject",
        face_crop: Optional[np.ndarray] = None,
        face_detected: bool = False,
    ) -> Optional[str]:
        """
        Process track for door crossing detection.

        Args:
            track_id: Track identifier
            track: TrackedObject with position info
            face_crop: Face crop if detected
            face_detected: Whether face was detected

        Returns:
            "entered", "exited", or None
        """
        center_y = track.center[1]
        current_zone = self._get_zone(center_y)
        prev_zone = self._track_states.get(track_id)

        # Update zone
        self._track_states[track_id] = current_zone

        # Detect crossing
        crossing_event = None

        if prev_zone == "above" and current_zone == "below":
            # Moved from above to below door line
            crossing_event = "entered" if self.enter_direction_down else "exited"
        elif prev_zone == "below" and current_zone == "above":
            # Moved from below to above door line
            crossing_event = "exited" if self.enter_direction_down else "entered"

        if crossing_event:
            logger.info(f"[CAPTURE_WORKER] Door crossing detected! track_id={track_id}, event={crossing_event}, prev_zone={prev_zone}, current_zone={current_zone}")
            self._on_door_crossing(
                track_id=track_id,
                track=track,
                event_type=crossing_event,
                face_crop=face_crop,
                face_detected=face_detected,
            )
        elif prev_zone != current_zone:
            logger.debug(f"[CAPTURE_WORKER] Track {track_id} zone change: {prev_zone} -> {current_zone} (y={center_y}, door_y={self.door_y})")

        return crossing_event

    def _get_zone(self, y: int) -> str:
        """Determine which zone a y-coordinate is in."""
        if y < self.door_y - self.crossing_threshold:
            return "above"
        elif y > self.door_y + self.crossing_threshold:
            return "below"
        else:
            return "crossing"

    def _on_door_crossing(
        self,
        track_id: int,
        track: "TrackedObject",
        event_type: str,
        face_crop: Optional[np.ndarray],
        face_detected: bool,
    ):
        """
        Handle door crossing event.

        1. Capture full body crop
        2. Log event immediately (even with identity=None)
        3. Queue for async recognition

        Args:
            track_id: Track identifier
            track: TrackedObject
            event_type: "entered" or "exited"
            face_crop: Face crop if available
            face_detected: Whether face was detected
        """
        timestamp = datetime.now().isoformat()

        # Capture body crop
        body_crop = self._capture_body_crop(track.bbox)
        if body_crop is None:
            logger.warning(f"Failed to capture body crop for track {track_id}")
            return

        # Create capture event
        capture_event = CaptureEvent(
            track_id=track_id,
            timestamp=timestamp,
            event_type="entry" if event_type == "entered" else "exit",
            full_body_crop=body_crop,
            face_crop=face_crop,
            face_detected=face_detected,
            box_count=track.box_count,
            processing_status="pending",
            door_crossing_logged=False,
        )

        # Log event immediately (deferred identity resolution)
        self._log_immediate_event(track_id, track, event_type)
        capture_event.door_crossing_logged = True

        # Queue for recognition
        if self.recognition_queue.put(capture_event):
            self.stats["captures_queued"] += 1
            queue_size = self.recognition_queue.size()
            logger.info(
                f"[CAPTURE_WORKER] Queued capture for track {track_id}: {event_type}, "
                f"face_detected={face_detected}, queue_size={queue_size}"
            )

        # Update stats
        if event_type == "entered":
            self.stats["entries_captured"] += 1
        else:
            self.stats["exits_captured"] += 1

    def _capture_body_crop(
        self,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract body crop from current frame.

        Args:
            bbox: Person bounding box (x1, y1, x2, y2)

        Returns:
            Body crop or None if failed
        """
        with self._frame_lock:
            if self._current_frame is None:
                return None

            frame = self._current_frame
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox

            # Add padding
            bw = x2 - x1
            bh = y2 - y1
            pad_x = int(bw * self.capture_padding)
            pad_y = int(bh * self.capture_padding)

            # Clamp to frame bounds
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            if x2 <= x1 or y2 <= y1:
                return None

            return frame[y1:y2, x1:x2].copy()

    def _log_immediate_event(
        self,
        track_id: int,
        track: "TrackedObject",
        event_type: str,
    ):
        """
        Log event immediately, even if identity is unknown.

        This enables <1s event logging while identity is resolved async.

        Args:
            track_id: Track identifier
            track: TrackedObject with current state
            event_type: "entered" or "exited"
        """
        identity = track.identity if track.identity != "Unknown" else None
        confidence = track.identity_confidence if identity else 0.0

        if event_type == "entered":
            self.event_manager.record_entry(
                track_id=track_id,
                identity=identity,
                identity_confidence=confidence,
                box_count=track.box_count,
                match_method=track.match_method,
                body_score=track.body_score,
            )
        else:
            # For exit, we need entry box count from context
            # This should be managed by the state machine
            self.event_manager.record_exit(
                track_id=track_id,
                identity=identity,
                identity_confidence=confidence,
                box_count=track.box_count,
                entry_box_count=0,  # Will be filled by state machine
                match_method=track.match_method,
                body_score=track.body_score,
            )

        logger.info(
            f"Immediate event logged: {event_type} track={track_id}, "
            f"identity={identity or 'pending'}"
        )

    def remove_track(self, track_id: int):
        """
        Clean up state for removed track.

        Args:
            track_id: Track to remove
        """
        self._track_states.pop(track_id, None)
        self._pending_captures.pop(track_id, None)

    def get_stats(self) -> Dict:
        """Get capture statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "entries_captured": 0,
            "exits_captured": 0,
            "captures_queued": 0,
        }
