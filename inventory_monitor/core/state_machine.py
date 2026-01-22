"""
Person state machine for tracking individual lifecycle in monitored area.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class PersonState(Enum):
    """States in person lifecycle."""
    DETECTED = 1        # Just detected, not yet tracked
    APPROACHING = 2     # Moving towards door
    AT_DOOR = 3         # At the door threshold
    ENTERING = 4        # Crossing door inward
    INSIDE = 5          # Inside the room
    EXITING = 6         # Crossing door outward
    LEFT = 7            # Left the monitored area


@dataclass
class PersonContext:
    """Context data for a tracked person."""
    track_id: int
    state: PersonState = PersonState.DETECTED
    identity: Optional[str] = None
    identity_confidence: float = 0.0
    identity_locked: bool = False

    # Timing
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None

    # Box tracking
    box_count: int = 0
    entry_box_count: int = 0
    exit_box_count: int = 0
    box_history: deque = field(default_factory=lambda: deque(maxlen=20))

    # Position tracking
    position_history: deque = field(default_factory=lambda: deque(maxlen=50))
    last_position: Optional[Tuple[int, int]] = None

    # State history
    state_history: List[Tuple[PersonState, datetime]] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.box_history, deque):
            self.box_history = deque(maxlen=20)
        if not isinstance(self.position_history, deque):
            self.position_history = deque(maxlen=50)
        self.state_history.append((self.state, datetime.now()))

    def transition_to(self, new_state: PersonState):
        """Transition to a new state."""
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            self.state_history.append((new_state, datetime.now()))
            logger.debug(f"Track {self.track_id}: {old_state.name} -> {new_state.name}")

    def update_position(self, center: Tuple[int, int]):
        """Update position history."""
        self.position_history.append(center)
        self.last_position = center
        self.last_seen = datetime.now()

    def update_boxes(self, count: int):
        """Update box count with smoothing."""
        self.box_history.append(count)
        # Use mode of recent counts for stability
        if len(self.box_history) >= 3:
            from collections import Counter
            recent = list(self.box_history)[-5:]
            most_common = Counter(recent).most_common(1)
            self.box_count = most_common[0][0]
        else:
            self.box_count = count

    @property
    def time_in_area(self) -> float:
        """Time spent in monitored area (seconds)."""
        return (self.last_seen - self.first_seen).total_seconds()

    @property
    def time_inside(self) -> Optional[float]:
        """Time spent inside the room (seconds)."""
        if self.entry_time is None:
            return None
        end_time = self.exit_time or datetime.now()
        return (end_time - self.entry_time).total_seconds()


class PersonStateMachine:
    """
    State machine manager for all tracked persons.

    Handles transitions based on:
    - Position relative to door
    - Movement direction
    - Time in each state
    """

    def __init__(
        self,
        door_y: int,
        door_threshold: int = 50,  # Pixels from door to trigger AT_DOOR
        enter_direction_down: bool = True,
    ):
        self.door_y = door_y
        self.door_threshold = door_threshold
        self.enter_direction_down = enter_direction_down

        self.persons: Dict[int, PersonContext] = {}

    def update(
        self,
        track_id: int,
        center: Tuple[int, int],
        box_count: int = 0,
        identity: Optional[str] = None,
        identity_confidence: float = 0.0,
    ) -> Optional[str]:
        """
        Update person state based on current position.

        Args:
            track_id: Person track ID
            center: Person center position
            box_count: Number of boxes carried
            identity: Recognized identity
            identity_confidence: Recognition confidence

        Returns:
            Event string if state transition triggers one
        """
        # Get or create context
        if track_id not in self.persons:
            self.persons[track_id] = PersonContext(track_id=track_id)

        ctx = self.persons[track_id]

        # Update tracking data
        ctx.update_position(center)
        ctx.update_boxes(box_count)

        # Update identity
        if identity and identity_confidence > ctx.identity_confidence:
            ctx.identity = identity
            ctx.identity_confidence = identity_confidence
            if identity_confidence >= 0.75:
                ctx.identity_locked = True

        # Calculate movement direction
        direction = self._get_direction(ctx)

        # State transitions
        event = self._process_state(ctx, center, direction)

        return event

    def _get_direction(self, ctx: PersonContext) -> Optional[str]:
        """Get movement direction from position history."""
        if len(ctx.position_history) < 5:
            return None

        positions = list(ctx.position_history)[-5:]
        y_delta = positions[-1][1] - positions[0][1]

        if abs(y_delta) < 15:
            return None

        if self.enter_direction_down:
            return "entering" if y_delta > 0 else "exiting"
        else:
            return "exiting" if y_delta > 0 else "entering"

    def _process_state(
        self,
        ctx: PersonContext,
        center: Tuple[int, int],
        direction: Optional[str]
    ) -> Optional[str]:
        """Process state transitions based on current state and position."""
        y = center[1]
        at_door = abs(y - self.door_y) < self.door_threshold

        event = None

        if ctx.state == PersonState.DETECTED:
            # Determine initial state based on position
            if self.enter_direction_down:
                if y < self.door_y:
                    ctx.transition_to(PersonState.APPROACHING)
                else:
                    ctx.transition_to(PersonState.INSIDE)
            else:
                if y > self.door_y:
                    ctx.transition_to(PersonState.APPROACHING)
                else:
                    ctx.transition_to(PersonState.INSIDE)

        elif ctx.state == PersonState.APPROACHING:
            if at_door:
                ctx.transition_to(PersonState.AT_DOOR)

        elif ctx.state == PersonState.AT_DOOR:
            if direction == "entering":
                ctx.transition_to(PersonState.ENTERING)
            elif direction == "exiting":
                ctx.transition_to(PersonState.EXITING)

        elif ctx.state == PersonState.ENTERING:
            # Check if crossed door
            crossed = (
                (self.enter_direction_down and y > self.door_y + self.door_threshold) or
                (not self.enter_direction_down and y < self.door_y - self.door_threshold)
            )
            if crossed:
                ctx.transition_to(PersonState.INSIDE)
                ctx.entry_time = datetime.now()
                ctx.entry_box_count = ctx.box_count
                event = "entered"
                logger.info(
                    f"Track {ctx.track_id} ({ctx.identity or 'Unknown'}) "
                    f"ENTERED with {ctx.box_count} boxes"
                )

        elif ctx.state == PersonState.INSIDE:
            if at_door and direction == "exiting":
                ctx.transition_to(PersonState.EXITING)

        elif ctx.state == PersonState.EXITING:
            # Check if crossed door
            crossed = (
                (self.enter_direction_down and y < self.door_y - self.door_threshold) or
                (not self.enter_direction_down and y > self.door_y + self.door_threshold)
            )
            if crossed:
                ctx.transition_to(PersonState.LEFT)
                ctx.exit_time = datetime.now()
                ctx.exit_box_count = ctx.box_count
                event = "exited"
                logger.info(
                    f"Track {ctx.track_id} ({ctx.identity or 'Unknown'}) "
                    f"EXITED with {ctx.box_count} boxes "
                    f"(entered with {ctx.entry_box_count})"
                )

        return event

    def get_context(self, track_id: int) -> Optional[PersonContext]:
        """Get person context."""
        return self.persons.get(track_id)

    def get_persons_inside(self) -> List[PersonContext]:
        """Get all persons currently inside."""
        return [
            ctx for ctx in self.persons.values()
            if ctx.state == PersonState.INSIDE
        ]

    def get_active_persons(self) -> Dict[int, PersonContext]:
        """Get all persons not yet left."""
        return {
            tid: ctx for tid, ctx in self.persons.items()
            if ctx.state != PersonState.LEFT
        }

    def remove_track(self, track_id: int):
        """Remove person from tracking."""
        if track_id in self.persons:
            del self.persons[track_id]

    def cleanup_left(self, max_age_seconds: float = 60.0):
        """Remove persons who left more than max_age_seconds ago."""
        now = datetime.now()
        to_remove = []

        for track_id, ctx in self.persons.items():
            if ctx.state == PersonState.LEFT:
                if (now - ctx.last_seen).total_seconds() > max_age_seconds:
                    to_remove.append(track_id)

        for track_id in to_remove:
            del self.persons[track_id]
