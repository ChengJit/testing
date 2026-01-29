"""Core modules for event handling and person state tracking."""

from .event_manager import EventManager, InventoryEvent, EventType
from .state_machine import PersonStateMachine, PersonState, PersonContext

__all__ = [
    "EventManager", "InventoryEvent", "EventType",
    "PersonStateMachine", "PersonState", "PersonContext",
]
