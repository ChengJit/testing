"""Core modules for event handling and person state tracking."""

from .event_manager import (
    EventManager, InventoryEvent, EventType, DeferredEvent,
    CaptureEvent, RecognitionResult
)
from .state_machine import PersonStateMachine, PersonState, PersonContext
from .recognition_worker import RecognitionQueue, RecognitionWorker
from .capture_worker import CaptureWorker

__all__ = [
    "EventManager", "InventoryEvent", "EventType", "DeferredEvent",
    "CaptureEvent", "RecognitionResult",
    "PersonStateMachine", "PersonState", "PersonContext",
    "RecognitionQueue", "RecognitionWorker", "CaptureWorker",
]
