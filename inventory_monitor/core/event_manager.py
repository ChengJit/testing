"""
Event management and logging for inventory monitoring.
Handles entry/exit events and generates reports.
"""

import logging
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Callable, TYPE_CHECKING
from pathlib import Path
from enum import Enum
import threading

if TYPE_CHECKING:
    from ..utils.api_client import CCTVAPIClient

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of inventory events."""
    PERSON_ENTERED = "entered"
    PERSON_EXITED = "exited"
    BOX_DETECTED = "box_detected"
    IDENTITY_RECOGNIZED = "identity_recognized"
    ALERT = "alert"


@dataclass
class InventoryEvent:
    """Single inventory event record."""
    timestamp: str
    event_type: EventType
    track_id: int
    identity: Optional[str]
    identity_confidence: float
    box_count: int
    direction: Optional[str]
    details: Dict

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


class EventManager:
    """
    Manages inventory events with persistence and callbacks.

    Features:
    - CSV logging for easy analysis
    - JSON event log for detailed records
    - Real-time callbacks for notifications
    - Daily log rotation
    - Summary statistics
    """

    def __init__(
        self,
        log_dir: str = "logs",
        csv_filename: str = "inventory_events.csv",
        json_filename: str = "events.jsonl",
        enable_csv: bool = True,
        enable_json: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.enable_csv = enable_csv
        self.enable_json = enable_json

        self.csv_path = self.log_dir / csv_filename
        self.json_path = self.log_dir / json_filename

        # Event storage
        self.events: List[InventoryEvent] = []
        self.callbacks: List[Callable[[InventoryEvent], None]] = []

        # Statistics
        self.stats = {
            "total_entries": 0,
            "total_exits": 0,
            "boxes_brought_in": 0,
            "boxes_taken_out": 0,
            "unique_persons": set(),
        }

        # Thread safety
        self._lock = threading.Lock()

        # Initialize CSV with headers
        if self.enable_csv:
            self._init_csv()

        logger.info(f"Event manager initialized, logging to {self.log_dir}")

    def _init_csv(self):
        """Initialize CSV file with headers if new."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "event_type",
                    "track_id",
                    "identity",
                    "identity_confidence",
                    "box_count",
                    "direction",
                    "details"
                ])

    def register_callback(self, callback: Callable[[InventoryEvent], None]):
        """Register callback to be called on each event."""
        self.callbacks.append(callback)

    def record_event(
        self,
        event_type: EventType,
        track_id: int,
        identity: Optional[str] = None,
        identity_confidence: float = 0.0,
        box_count: int = 0,
        direction: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> InventoryEvent:
        """
        Record a new inventory event.

        Args:
            event_type: Type of event
            track_id: Person track ID
            identity: Recognized identity (if any)
            identity_confidence: Recognition confidence
            box_count: Number of boxes
            direction: Movement direction
            details: Additional event details

        Returns:
            Created InventoryEvent
        """
        event = InventoryEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            track_id=track_id,
            identity=identity,
            identity_confidence=identity_confidence,
            box_count=box_count,
            direction=direction,
            details=details or {},
        )

        with self._lock:
            # Store in memory
            self.events.append(event)

            # Update statistics
            self._update_stats(event)

            # Persist to files
            if self.enable_csv:
                self._write_csv(event)
            if self.enable_json:
                self._write_json(event)

        # Trigger callbacks
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

        logger.info(
            f"Event: {event_type.value} | Track: {track_id} | "
            f"Identity: {identity} | Boxes: {box_count}"
        )

        return event

    def _update_stats(self, event: InventoryEvent):
        """Update running statistics."""
        if event.event_type == EventType.PERSON_ENTERED:
            self.stats["total_entries"] += 1
            self.stats["boxes_brought_in"] += event.box_count
        elif event.event_type == EventType.PERSON_EXITED:
            self.stats["total_exits"] += 1
            self.stats["boxes_taken_out"] += event.box_count

        if event.identity and event.identity != "Unknown":
            self.stats["unique_persons"].add(event.identity)

    def _write_csv(self, event: InventoryEvent):
        """Append event to CSV file."""
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    event.timestamp,
                    event.event_type.value,
                    event.track_id,
                    event.identity or "",
                    f"{event.identity_confidence:.2f}",
                    event.box_count,
                    event.direction or "",
                    json.dumps(event.details),
                ])
        except Exception as e:
            logger.error(f"Failed to write CSV: {e}")

    def _write_json(self, event: InventoryEvent):
        """Append event to JSON lines file."""
        try:
            with open(self.json_path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write JSON: {e}")

    def record_entry(
        self,
        track_id: int,
        identity: Optional[str] = None,
        identity_confidence: float = 0.0,
        box_count: int = 0,
    ) -> InventoryEvent:
        """Convenience method for recording person entry."""
        return self.record_event(
            event_type=EventType.PERSON_ENTERED,
            track_id=track_id,
            identity=identity,
            identity_confidence=identity_confidence,
            box_count=box_count,
            direction="entering",
        )

    def record_exit(
        self,
        track_id: int,
        identity: Optional[str] = None,
        identity_confidence: float = 0.0,
        box_count: int = 0,
        entry_box_count: int = 0,
    ) -> InventoryEvent:
        """Convenience method for recording person exit."""
        box_diff = box_count - entry_box_count
        details = {
            "entry_box_count": entry_box_count,
            "exit_box_count": box_count,
            "box_difference": box_diff,
        }

        # Alert if taking more boxes than brought in
        if box_diff > 0:
            details["alert"] = "Taking more boxes than brought in"

        return self.record_event(
            event_type=EventType.PERSON_EXITED,
            track_id=track_id,
            identity=identity,
            identity_confidence=identity_confidence,
            box_count=box_count,
            direction="exiting",
            details=details,
        )

    def get_statistics(self) -> Dict:
        """Get current statistics."""
        with self._lock:
            return {
                "total_entries": self.stats["total_entries"],
                "total_exits": self.stats["total_exits"],
                "boxes_brought_in": self.stats["boxes_brought_in"],
                "boxes_taken_out": self.stats["boxes_taken_out"],
                "net_box_flow": self.stats["boxes_taken_out"] - self.stats["boxes_brought_in"],
                "unique_persons": len(self.stats["unique_persons"]),
                "persons_currently_inside": self.stats["total_entries"] - self.stats["total_exits"],
            }

    def get_recent_events(self, count: int = 10) -> List[InventoryEvent]:
        """Get most recent events."""
        with self._lock:
            return list(self.events[-count:])

    def get_events_by_track(self, track_id: int) -> List[InventoryEvent]:
        """Get all events for a specific track."""
        with self._lock:
            return [e for e in self.events if e.track_id == track_id]

    def get_events_by_identity(self, identity: str) -> List[InventoryEvent]:
        """Get all events for a specific person."""
        with self._lock:
            return [e for e in self.events if e.identity == identity]

    def generate_daily_report(self) -> Dict:
        """Generate daily summary report."""
        today = datetime.now().date().isoformat()

        with self._lock:
            today_events = [
                e for e in self.events
                if e.timestamp.startswith(today)
            ]

        entries = [e for e in today_events if e.event_type == EventType.PERSON_ENTERED]
        exits = [e for e in today_events if e.event_type == EventType.PERSON_EXITED]

        report = {
            "date": today,
            "total_entries": len(entries),
            "total_exits": len(exits),
            "boxes_brought_in": sum(e.box_count for e in entries),
            "boxes_taken_out": sum(e.box_count for e in exits),
            "unique_visitors": len(set(e.identity for e in today_events if e.identity)),
            "events": [e.to_dict() for e in today_events],
        }

        # Save report
        report_path = self.log_dir / f"daily_report_{today}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Daily report generated: {report_path}")
        return report

    # ========== API Integration ==========

    def enable_api_reporting(
        self,
        api_url: str = "https://ops-portal.fasspay.com/cctv",
        camera_id: str = "cam-001",
        verify_ssl: bool = True,
        capture_frame_callback: Optional[Callable[[], Optional[bytes]]] = None,
        heartbeat_interval: int = 30,
    ) -> "CCTVAPIClient":
        """
        Enable sending events to the remote API.

        Args:
            api_url: Base URL of the kafka-report API
            camera_id: Unique identifier for this camera
            verify_ssl: Whether to verify SSL certificates
            capture_frame_callback: Optional callback to get current frame as JPEG bytes
            heartbeat_interval: Seconds between heartbeats (default 30)

        Returns:
            The initialized CCTVAPIClient instance
        """
        from ..utils.api_client import CCTVAPIClient

        self._api_client = CCTVAPIClient(
            base_url=api_url,
            camera_id=camera_id,
            verify_ssl=verify_ssl,
            async_mode=True,
        )

        self._capture_frame_callback = capture_frame_callback

        # Register callback to send events to API
        self.register_callback(self._api_event_callback)

        # Start heartbeat to indicate monitor is online
        self._api_client.start_heartbeat(interval=heartbeat_interval)

        logger.info(f"API reporting enabled: {api_url} (camera: {camera_id})")
        return self._api_client

    def _api_event_callback(self, event: InventoryEvent):
        """Callback to send events to the remote API."""
        if not hasattr(self, '_api_client') or self._api_client is None:
            return

        # Get frame image if callback provided
        image_bytes = None
        if hasattr(self, '_capture_frame_callback') and self._capture_frame_callback:
            try:
                image_bytes = self._capture_frame_callback()
            except Exception as e:
                logger.warning(f"Failed to capture frame for API: {e}")

        # Send to API
        try:
            self._api_client.send_event(
                event_type=event.event_type.value,
                identity=event.identity,
                identity_confidence=event.identity_confidence,
                box_count=event.box_count,
                direction=event.direction,
                track_id=event.track_id,
                details=event.details,
                image=image_bytes,
            )
        except Exception as e:
            logger.error(f"Failed to send event to API: {e}")

    def disable_api_reporting(self):
        """Disable API reporting and cleanup."""
        if hasattr(self, '_api_client') and self._api_client:
            self._api_client.stop()
            self._api_client = None
            logger.info("API reporting disabled")

    def get_api_client(self) -> Optional["CCTVAPIClient"]:
        """Get the current API client instance."""
        return getattr(self, '_api_client', None)
