"""
Zone management for entry/exit detection.
Handles door line crossing and directional movement tracking.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class Zone(Enum):
    """Defined zones in the monitored area."""
    OUTSIDE = 1      # Before entry zone (outside room)
    ENTRY = 2        # Entry zone near door
    INSIDE = 3       # Inside the room
    EXIT = 4         # Exit zone near door


@dataclass
class PersonZoneState:
    """Tracks a person's zone state and transitions."""
    track_id: int
    current_zone: Zone = Zone.OUTSIDE
    previous_zone: Optional[Zone] = None

    # Position tracking
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    zone_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Direction confirmation
    direction_votes: deque = field(default_factory=lambda: deque(maxlen=10))
    confirmed_direction: Optional[str] = None  # "entering" or "exiting"

    # Entry/exit state
    has_entered: bool = False
    has_exited: bool = False
    entry_box_count: int = 0
    exit_box_count: int = 0

    def __post_init__(self):
        if not isinstance(self.position_history, deque):
            self.position_history = deque(maxlen=30)
        if not isinstance(self.zone_history, deque):
            self.zone_history = deque(maxlen=10)
        if not isinstance(self.direction_votes, deque):
            self.direction_votes = deque(maxlen=10)


class ZoneManager:
    """
    Manages zone definitions and person state transitions.

    The door is assumed to be at a specific line (door_line).
    Entry/exit zones are defined around this line.
    """

    def __init__(
        self,
        frame_height: int,
        frame_width: int,
        door_line: float = 0.5,         # Normalized position (0=top, 1=bottom)
        entry_zone_size: float = 0.15,  # Size of entry/exit zones
        enter_direction_down: bool = True,  # True = entering when moving down
        direction_frames: int = 5,      # Frames to confirm direction
        min_movement: int = 20,         # Minimum pixels for movement
    ):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.enter_direction_down = enter_direction_down
        self.direction_frames = direction_frames
        self.min_movement = min_movement

        # Calculate zone boundaries (in pixels)
        door_y = int(door_line * frame_height)
        zone_size = int(entry_zone_size * frame_height)

        self.door_line = door_y
        self.entry_zone_top = door_y - zone_size
        self.entry_zone_bottom = door_y
        self.exit_zone_top = door_y
        self.exit_zone_bottom = door_y + zone_size

        # Person states
        self.person_states: Dict[int, PersonZoneState] = {}

        logger.info(
            f"Zone manager initialized: door_line={door_y}, "
            f"entry=[{self.entry_zone_top}, {self.entry_zone_bottom}], "
            f"exit=[{self.exit_zone_top}, {self.exit_zone_bottom}]"
        )

    def update(
        self,
        track_id: int,
        center: Tuple[int, int],
        box_count: int = 0
    ) -> Optional[str]:
        """
        Update person's zone state and detect transitions.

        Args:
            track_id: Person track ID
            center: Person center position (x, y)
            box_count: Number of boxes being carried

        Returns:
            Event string if transition detected: "entered" or "exited"
        """
        # Get or create state
        if track_id not in self.person_states:
            self.person_states[track_id] = PersonZoneState(track_id=track_id)

        state = self.person_states[track_id]

        # Update position history
        state.position_history.append(center)

        # Determine current zone
        y = center[1]
        new_zone = self._get_zone(y)

        # Track zone changes
        if new_zone != state.current_zone:
            state.previous_zone = state.current_zone
            state.current_zone = new_zone
            state.zone_history.append(new_zone)

        # Update direction
        direction = self._calculate_direction(state)
        if direction:
            state.direction_votes.append(direction)
            state.confirmed_direction = self._confirm_direction(state)

        # Detect entry/exit events
        event = self._detect_transition(state, box_count)

        return event

    def _get_zone(self, y: int) -> Zone:
        """Determine which zone a y-coordinate is in."""
        if y < self.entry_zone_top:
            return Zone.OUTSIDE
        elif y < self.entry_zone_bottom:
            return Zone.ENTRY
        elif y < self.exit_zone_bottom:
            return Zone.INSIDE if y >= self.exit_zone_top else Zone.ENTRY
        else:
            return Zone.INSIDE  # Past exit zone = inside room

    def _calculate_direction(self, state: PersonZoneState) -> Optional[str]:
        """Calculate movement direction from recent positions."""
        if len(state.position_history) < 3:
            return None

        positions = list(state.position_history)[-5:]
        if len(positions) < 3:
            return None

        y_delta = positions[-1][1] - positions[0][1]

        if abs(y_delta) < self.min_movement:
            return None

        # Positive y delta = moving down
        if self.enter_direction_down:
            return "entering" if y_delta > 0 else "exiting"
        else:
            return "exiting" if y_delta > 0 else "entering"

    def _confirm_direction(self, state: PersonZoneState) -> Optional[str]:
        """Confirm direction based on vote majority."""
        if len(state.direction_votes) < self.direction_frames:
            return None

        recent_votes = list(state.direction_votes)[-self.direction_frames:]
        entering_count = sum(1 for v in recent_votes if v == "entering")
        exiting_count = sum(1 for v in recent_votes if v == "exiting")

        if entering_count >= self.direction_frames - 1:
            return "entering"
        elif exiting_count >= self.direction_frames - 1:
            return "exiting"

        return None

    def _detect_transition(
        self,
        state: PersonZoneState,
        box_count: int
    ) -> Optional[str]:
        """Detect entry/exit transition events."""
        # Need confirmed direction
        if not state.confirmed_direction:
            return None

        # Entry detection: moving from OUTSIDE/ENTRY to INSIDE
        if state.confirmed_direction == "entering":
            if not state.has_entered:
                if state.current_zone == Zone.INSIDE and state.previous_zone in (Zone.ENTRY, Zone.OUTSIDE):
                    state.has_entered = True
                    state.entry_box_count = box_count
                    logger.info(f"Track {state.track_id} ENTERED with {box_count} boxes")
                    return "entered"

        # Exit detection: moving from INSIDE to ENTRY/OUTSIDE
        elif state.confirmed_direction == "exiting":
            if state.has_entered and not state.has_exited:
                if state.current_zone in (Zone.ENTRY, Zone.OUTSIDE) and state.previous_zone == Zone.INSIDE:
                    state.has_exited = True
                    state.exit_box_count = box_count
                    logger.info(f"Track {state.track_id} EXITED with {box_count} boxes")
                    return "exited"

        return None

    def get_state(self, track_id: int) -> Optional[PersonZoneState]:
        """Get person's zone state."""
        return self.person_states.get(track_id)

    def remove_track(self, track_id: int):
        """Remove track from zone manager."""
        if track_id in self.person_states:
            del self.person_states[track_id]

    def cleanup_stale(self, active_track_ids: List[int]):
        """Remove states for tracks that no longer exist."""
        stale = [tid for tid in self.person_states if tid not in active_track_ids]
        for tid in stale:
            del self.person_states[tid]

    def get_zone_info(self) -> Dict:
        """Get zone boundary information for visualization."""
        return {
            "door_line": self.door_line,
            "entry_zone": (self.entry_zone_top, self.entry_zone_bottom),
            "exit_zone": (self.exit_zone_top, self.exit_zone_bottom),
        }
