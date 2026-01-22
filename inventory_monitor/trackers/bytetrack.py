"""
ByteTrack-inspired multi-object tracker optimized for person tracking.
Provides robust tracking even with occlusions and re-identification.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class TrackState(Enum):
    """Track lifecycle states."""
    TENTATIVE = 1   # New track, not yet confirmed
    CONFIRMED = 2   # Confirmed track with enough hits
    LOST = 3        # Lost track, searching


@dataclass
class TrackedObject:
    """
    Tracked person/object with state and history.
    """
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    state: TrackState = TrackState.TENTATIVE

    # Identity
    identity: Optional[str] = None
    identity_confidence: float = 0.0
    identity_locked: bool = False

    # Tracking state
    hits: int = 1
    age: int = 0
    time_since_update: int = 0

    # Position history for direction tracking
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))

    # Box tracking
    box_count: int = 0
    box_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Kalman filter state (simplified)
    velocity: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        if not isinstance(self.position_history, deque):
            self.position_history = deque(maxlen=30)
        if not isinstance(self.box_history, deque):
            self.box_history = deque(maxlen=10)
        self._update_position_history()

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def _update_position_history(self):
        """Add current center to position history."""
        self.position_history.append(self.center)

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float
    ):
        """Update track with new detection."""
        # Calculate velocity
        old_center = self.center
        self.bbox = bbox
        new_center = self.center
        self.velocity = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        )

        self.confidence = confidence
        self.hits += 1
        self.age += 1
        self.time_since_update = 0

        # Update state
        if self.state == TrackState.TENTATIVE and self.hits >= 3:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

        self._update_position_history()

    def mark_missed(self):
        """Mark track as missed this frame."""
        self.age += 1
        self.time_since_update += 1

        # Predict position using velocity
        x1, y1, x2, y2 = self.bbox
        vx, vy = self.velocity
        self.bbox = (
            int(x1 + vx),
            int(y1 + vy),
            int(x2 + vx),
            int(y2 + vy)
        )

        if self.state == TrackState.CONFIRMED and self.time_since_update > 5:
            self.state = TrackState.LOST

    def set_identity(self, name: str, confidence: float, lock: bool = False):
        """Set or update identity."""
        if self.identity_locked:
            return

        if confidence > self.identity_confidence:
            self.identity = name
            self.identity_confidence = confidence

            if lock or confidence >= 0.75:
                self.identity_locked = True
                logger.info(f"Track {self.track_id} identity locked: {name}")

    def update_box_count(self, count: int):
        """Update box count with temporal smoothing."""
        self.box_history.append(count)

        # Median filter for stability
        if len(self.box_history) >= 3:
            self.box_count = int(np.median(list(self.box_history)))
        else:
            self.box_count = count

    def get_direction(self, frame_count: int = 5) -> Optional[str]:
        """
        Get movement direction based on position history.

        Returns:
            "entering", "exiting", or None if unclear
        """
        if len(self.position_history) < frame_count:
            return None

        positions = list(self.position_history)[-frame_count:]
        y_values = [p[1] for p in positions]

        # Calculate overall y movement
        y_delta = y_values[-1] - y_values[0]

        # Threshold for significant movement
        if abs(y_delta) < 15:
            return None

        # Positive y = moving down = entering (camera facing door)
        return "entering" if y_delta > 0 else "exiting"


class ByteTracker:
    """
    ByteTrack-inspired multi-object tracker.

    Features:
    - Two-stage association (high/low confidence)
    - Robust handling of occlusions
    - Track lifecycle management
    - Velocity-based prediction
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_hits: int = 3,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_hits = min_hits

        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.frame_count = 0

    def update(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], float]]
    ) -> Dict[int, TrackedObject]:
        """
        Update tracks with new detections.

        Args:
            detections: List of (bbox, confidence) tuples

        Returns:
            Dictionary of active tracks
        """
        self.frame_count += 1

        if not detections:
            # Mark all tracks as missed
            self._mark_all_missed()
            return self._get_active_tracks()

        # Separate high and low confidence detections
        high_dets = [(b, c) for b, c in detections if c >= self.track_thresh]
        low_dets = [(b, c) for b, c in detections if c < self.track_thresh]

        # Get active tracks for matching
        confirmed_tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.state == TrackState.CONFIRMED
        }
        tentative_tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.state == TrackState.TENTATIVE
        }

        # First association: high confidence detections with confirmed tracks
        matched_tracks, unmatched_tracks, unmatched_dets = self._associate(
            confirmed_tracks, high_dets, self.match_thresh
        )

        # Update matched tracks
        for track_id, det_idx in matched_tracks:
            bbox, conf = high_dets[det_idx]
            self.tracks[track_id].update(bbox, conf)

        # Second association: remaining tracks with low confidence detections
        remaining_tracks = {tid: self.tracks[tid] for tid in unmatched_tracks}
        matched_2, unmatched_tracks_2, unmatched_low = self._associate(
            remaining_tracks, low_dets, self.match_thresh * 0.8
        )

        for track_id, det_idx in matched_2:
            bbox, conf = low_dets[det_idx]
            self.tracks[track_id].update(bbox, conf)

        # Third association: tentative tracks with remaining high confidence detections
        remaining_high_dets = [high_dets[i] for i in unmatched_dets]
        matched_3, _, remaining_dets = self._associate(
            tentative_tracks, remaining_high_dets, self.match_thresh * 0.9
        )

        for track_id, det_idx in matched_3:
            bbox, conf = remaining_high_dets[det_idx]
            self.tracks[track_id].update(bbox, conf)

        # Mark unmatched tracks as missed
        all_unmatched = set(unmatched_tracks_2) | set(tentative_tracks.keys()) - {
            t[0] for t in matched_3
        }
        for track_id in all_unmatched:
            if track_id in self.tracks:
                self.tracks[track_id].mark_missed()

        # Create new tracks for unmatched high confidence detections
        for det_idx in remaining_dets:
            bbox, conf = remaining_high_dets[det_idx]
            self._create_track(bbox, conf)

        # Remove dead tracks
        self._remove_dead_tracks()

        return self._get_active_tracks()

    def _associate(
        self,
        tracks: Dict[int, TrackedObject],
        detections: List[Tuple[Tuple[int, int, int, int], float]],
        thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using IoU.

        Returns:
            matched: List of (track_id, detection_idx) tuples
            unmatched_tracks: List of unmatched track IDs
            unmatched_dets: List of unmatched detection indices
        """
        if not tracks or not detections:
            return [], list(tracks.keys()), list(range(len(detections)))

        # Calculate IoU cost matrix
        track_ids = list(tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))

        for i, track_id in enumerate(track_ids):
            track_bbox = tracks[track_id].bbox
            for j, (det_bbox, _) in enumerate(detections):
                cost_matrix[i, j] = 1 - self._iou(track_bbox, det_bbox)

        # Hungarian algorithm for assignment
        matched, unmatched_tracks, unmatched_dets = self._linear_assignment(
            cost_matrix, track_ids, thresh
        )

        return matched, unmatched_tracks, unmatched_dets

    def _linear_assignment(
        self,
        cost_matrix: np.ndarray,
        track_ids: List[int],
        thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Simple greedy assignment (could use scipy.optimize.linear_sum_assignment)."""
        matched = []
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(cost_matrix.shape[1]))

        # Greedy matching by minimum cost
        while True:
            if not unmatched_tracks or not unmatched_dets:
                break

            # Find minimum cost
            min_cost = float('inf')
            best_match = None

            for i, track_id in enumerate(track_ids):
                if track_id not in unmatched_tracks:
                    continue
                for j in unmatched_dets:
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        best_match = (track_id, j)

            # Check threshold
            if best_match is None or min_cost > (1 - thresh):
                break

            matched.append(best_match)
            unmatched_tracks.discard(best_match[0])
            unmatched_dets.discard(best_match[1])

        return matched, list(unmatched_tracks), list(unmatched_dets)

    def _create_track(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float
    ) -> TrackedObject:
        """Create new track."""
        track = TrackedObject(
            track_id=self.next_id,
            bbox=bbox,
            confidence=confidence,
        )
        self.tracks[self.next_id] = track
        self.next_id += 1
        logger.debug(f"Created track {track.track_id}")
        return track

    def _mark_all_missed(self):
        """Mark all tracks as missed."""
        for track in self.tracks.values():
            track.mark_missed()

    def _remove_dead_tracks(self):
        """Remove tracks that have been lost for too long."""
        to_remove = []

        for track_id, track in self.tracks.items():
            # Remove tentative tracks that haven't been confirmed
            if track.state == TrackState.TENTATIVE and track.time_since_update > 3:
                to_remove.append(track_id)
            # Remove lost tracks that have exceeded buffer
            elif track.state == TrackState.LOST and track.time_since_update > self.track_buffer:
                to_remove.append(track_id)

        for track_id in to_remove:
            logger.debug(f"Removed track {track_id}")
            del self.tracks[track_id]

    def _get_active_tracks(self) -> Dict[int, TrackedObject]:
        """Get currently active (confirmed or tentative) tracks."""
        return {
            tid: t for tid, t in self.tracks.items()
            if t.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)
            and t.time_since_update <= 5
        }

    @staticmethod
    def _iou(
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Get specific track by ID."""
        return self.tracks.get(track_id)

    def get_confirmed_tracks(self) -> Dict[int, TrackedObject]:
        """Get only confirmed tracks."""
        return {
            tid: t for tid, t in self.tracks.items()
            if t.state == TrackState.CONFIRMED
        }
