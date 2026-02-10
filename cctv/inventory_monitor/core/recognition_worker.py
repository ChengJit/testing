"""
Recognition Worker for asynchronous identity resolution.

Processes captured events in a background thread, trying face recognition
first, then falling back to body/clothing matching.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue, Empty, Full
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

import numpy as np

from .event_manager import CaptureEvent, RecognitionResult, EventType

if TYPE_CHECKING:
    from ..detectors.face import FaceRecognizer
    from ..detectors.body import BodyRecognizer
    from .event_manager import EventManager

logger = logging.getLogger(__name__)


class RecognitionQueue:
    """
    Thread-safe queue for capture events awaiting recognition.

    Features:
    - Fixed max size with drop-oldest policy
    - Thread-safe operations
    - Priority support (optional)
    """

    def __init__(self, max_size: int = 20):
        """
        Initialize recognition queue.

        Args:
            max_size: Maximum queue size. When full, oldest items are dropped.
        """
        self.max_size = max_size
        self._queue: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def put(self, event: CaptureEvent) -> bool:
        """
        Add event to queue. Drops oldest if full.

        Args:
            event: CaptureEvent to enqueue

        Returns:
            True if added, False if queue was full (oldest was dropped)
        """
        with self._lock:
            was_full = len(self._queue) >= self.max_size
            if was_full:
                dropped = self._queue.popleft()
                logger.warning(
                    f"Recognition queue full, dropped event for track {dropped.track_id}"
                )

            self._queue.append(event)
            self._not_empty.notify()
            return not was_full

    def get(self, timeout: float = 1.0) -> Optional[CaptureEvent]:
        """
        Get next event from queue.

        Args:
            timeout: Max seconds to wait

        Returns:
            CaptureEvent or None if timeout
        """
        with self._not_empty:
            if not self._queue:
                self._not_empty.wait(timeout)

            if self._queue:
                return self._queue.popleft()
            return None

    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)

    def clear(self):
        """Clear all items from queue."""
        with self._lock:
            self._queue.clear()

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0


class RecognitionWorker:
    """
    Background worker for processing recognition queue.

    Processes CaptureEvents asynchronously:
    1. Try face recognition first (highest confidence)
    2. Fall back to body recognition if face fails
    3. Try closest match as last resort
    4. Send IDENTITY_RESOLVED event when identity found
    """

    def __init__(
        self,
        face_recognizer: "FaceRecognizer",
        body_recognizer: Optional["BodyRecognizer"],
        event_manager: "EventManager",
        queue: Optional[RecognitionQueue] = None,
        camera_id: str = "cam-001",
        face_threshold: float = 0.45,
        body_threshold: float = 0.6,
        closest_threshold: float = 0.5,
    ):
        """
        Initialize recognition worker.

        Args:
            face_recognizer: Face recognition module
            body_recognizer: Body recognition module (optional)
            event_manager: Event manager for recording results
            queue: Recognition queue (created if not provided)
            camera_id: Camera identifier for events
            face_threshold: Minimum face recognition score
            body_threshold: Minimum body recognition score
            closest_threshold: Minimum closest-match score
        """
        self.face_recognizer = face_recognizer
        self.body_recognizer = body_recognizer
        self.event_manager = event_manager
        self.queue = queue or RecognitionQueue()
        self.camera_id = camera_id
        self.face_threshold = face_threshold
        self.body_threshold = body_threshold
        self.closest_threshold = closest_threshold

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Statistics
        self.stats = {
            "processed": 0,
            "face_matches": 0,
            "body_matches": 0,
            "closest_matches": 0,
            "unresolved": 0,
        }

    def start(self):
        """Start the recognition worker thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True, name="RecognitionWorker")
        self._thread.start()
        logger.info(f"[RECOGNITION_WORKER] Started - face_thresh={self.face_threshold}, body_thresh={self.body_threshold}, closest_thresh={self.closest_threshold}")

    def stop(self):
        """Stop the recognition worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info(f"Recognition worker stopped. Stats: {self.stats}")

    def enqueue(self, event: CaptureEvent) -> bool:
        """Add capture event to processing queue."""
        return self.queue.put(event)

    def _process_loop(self):
        """Main processing loop."""
        logger.info("[RECOGNITION_WORKER] Processing loop started")
        loop_count = 0
        while self._running:
            loop_count += 1
            queue_size = self.queue.size()

            if loop_count % 100 == 0:
                logger.debug(f"[RECOGNITION_WORKER] Loop #{loop_count}, queue_size={queue_size}, stats={self.stats}")

            event = self.queue.get(timeout=0.5)
            if event is None:
                continue

            logger.info(f"[RECOGNITION_WORKER] Processing event: track_id={event.track_id}, type={event.event_type}, face_detected={event.face_detected}")

            try:
                result = self._process_event(event)
                logger.info(f"[RECOGNITION_WORKER] Result: identity={result.identity}, method={result.match_method}, confidence={result.confidence:.2f}")
                if result.identity:
                    self._send_identity_resolved(event, result)
            except Exception as e:
                logger.error(f"[RECOGNITION_WORKER] Processing error: {e}", exc_info=True)

        logger.info("[RECOGNITION_WORKER] Processing loop ended")

    def _process_event(self, event: CaptureEvent) -> RecognitionResult:
        """
        Process a single capture event through recognition pipeline.

        Order: face -> body -> closest -> unknown

        Args:
            event: CaptureEvent to process

        Returns:
            RecognitionResult with best match
        """
        self.stats["processed"] += 1
        event.processing_status = "processing"

        result = RecognitionResult(track_id=event.track_id)

        # 1. Try face recognition first (highest priority)
        if event.face_detected and event.face_crop is not None:
            face_result = self._try_face_recognition(event.face_crop, event.track_id)
            if face_result:
                result.identity = face_result[0]
                result.confidence = face_result[1]
                result.face_score = face_result[1]
                result.match_method = "face"
                self.stats["face_matches"] += 1
                event.processing_status = "resolved"
                logger.debug(
                    f"Face match for track {event.track_id}: "
                    f"{result.identity} ({result.confidence:.2f})"
                )
                return result

        # 2. Try body recognition
        if self.body_recognizer and event.full_body_crop is not None:
            body_result = self.body_recognizer.recognize(event.full_body_crop)
            if body_result.name and body_result.confidence >= self.body_threshold:
                result.identity = body_result.name
                result.confidence = body_result.confidence
                result.body_score = body_result.confidence
                result.match_method = "body"
                result.needs_review = body_result.needs_review
                self.stats["body_matches"] += 1
                event.processing_status = "resolved"
                logger.debug(
                    f"Body match for track {event.track_id}: "
                    f"{result.identity} ({result.confidence:.2f})"
                )
                return result

        # 3. Try closest match fallback
        if self.body_recognizer and event.full_body_crop is not None:
            closest = self.body_recognizer.find_closest(event.full_body_crop)
            if closest.name and closest.confidence >= self.closest_threshold:
                result.identity = closest.name
                result.confidence = closest.confidence
                result.body_score = closest.confidence
                result.match_method = "closest"
                result.needs_review = True  # Always review closest matches
                self.stats["closest_matches"] += 1
                event.processing_status = "resolved"
                logger.debug(
                    f"Closest match for track {event.track_id}: "
                    f"{result.identity} ({result.confidence:.2f})"
                )
                return result

        # 4. No match found
        self.stats["unresolved"] += 1
        event.processing_status = "unresolved"
        logger.debug(f"No match found for track {event.track_id}")

        return result

    def _try_face_recognition(
        self,
        face_crop: np.ndarray,
        track_id: int
    ) -> Optional[tuple]:
        """
        Attempt face recognition on crop.

        Args:
            face_crop: Face image crop
            track_id: Track identifier

        Returns:
            (name, confidence) tuple or None
        """
        try:
            # Use face recognizer's recognize method with crop
            # Note: This assumes face_recognizer has a method to process crops
            result = self.face_recognizer.recognize_crop(face_crop, track_id)
            if result and result.name != "Unknown":
                if result.confidence >= self.face_threshold:
                    return (result.name, result.confidence)
        except AttributeError:
            # recognize_crop not available, skip face recognition
            logger.debug("Face recognizer does not support crop recognition")
        except Exception as e:
            logger.debug(f"Face recognition failed: {e}")

        return None

    def _send_identity_resolved(
        self,
        event: CaptureEvent,
        result: RecognitionResult
    ):
        """
        Send IDENTITY_RESOLVED event to event manager.

        Args:
            event: Original capture event
            result: Recognition result
        """
        self.event_manager.record_identity_resolved(
            track_id=event.track_id,
            identity=result.identity,
            identity_confidence=result.confidence,
            camera_id=self.camera_id,
        )

        logger.info(
            f"Identity resolved: track={event.track_id}, "
            f"identity={result.identity}, method={result.match_method}, "
            f"confidence={result.confidence:.2f}"
        )

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.queue.size()
