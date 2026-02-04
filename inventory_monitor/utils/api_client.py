"""
API Client for sending CCTV events to kafka-report service.

This module provides a client for sending entry/exit events and images
to the ops-portal kafka-report service for centralized logging.
"""

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Dict, Any, List, Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class CCTVEvent:
    """Data class representing a CCTV event."""
    timestamp: str
    event_type: str  # 'entered' or 'exited'
    track_id: Optional[int] = None
    identity: Optional[str] = None
    identity_confidence: Optional[float] = None
    box_count: int = 0
    direction: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    camera_id: Optional[str] = None
    image_base64: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


class CCTVAPIClient:
    """
    Client for sending CCTV events to the kafka-report API.

    Features:
    - Automatic retry with exponential backoff
    - Background queue for async sending
    - Batch upload support
    - Connection pooling
    - SSL verification configurable
    """

    def __init__(
        self,
        base_url: str = "https://ops-portal.fasspay.com/report",
        camera_id: str = "cam-001",
        verify_ssl: bool = True,
        timeout: int = 30,
        max_retries: int = 3,
        batch_size: int = 10,
        batch_interval: float = 5.0,
        async_mode: bool = True
    ):
        """
        Initialize the CCTV API client.

        Args:
            base_url: Base URL of the kafka-report service
            camera_id: Unique identifier for this camera
            verify_ssl: Whether to verify SSL certificates
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            batch_size: Number of events to batch before sending
            batch_interval: Max seconds to wait before sending partial batch
            async_mode: If True, events are queued and sent in background
        """
        self.base_url = base_url.rstrip('/')
        self.camera_id = camera_id
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.async_mode = async_mode

        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Background queue for async mode
        self._event_queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pending_batch: List[Dict] = []
        self._last_batch_time = time.time()

        # Callbacks
        self._on_send_success: Optional[Callable] = None
        self._on_send_error: Optional[Callable] = None

        if async_mode:
            self._start_worker()

    def _start_worker(self):
        """Start background worker thread for async sending."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("CCTV API client worker started")

    def _worker_loop(self):
        """Background worker loop for processing queued events."""
        while not self._stop_event.is_set():
            try:
                # Try to get event from queue with timeout
                try:
                    event_dict = self._event_queue.get(timeout=1.0)
                    self._pending_batch.append(event_dict)
                except Empty:
                    pass

                # Check if we should send batch
                should_send = (
                    len(self._pending_batch) >= self.batch_size or
                    (self._pending_batch and
                     time.time() - self._last_batch_time >= self.batch_interval)
                )

                if should_send and self._pending_batch:
                    self._send_batch(self._pending_batch.copy())
                    self._pending_batch.clear()
                    self._last_batch_time = time.time()

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(1)

        # Send remaining events on shutdown
        if self._pending_batch:
            self._send_batch(self._pending_batch)

    def _send_batch(self, events: List[Dict]) -> bool:
        """Send a batch of events to the API."""
        if not events:
            return True

        if len(events) == 1:
            return self._send_single(events[0])

        try:
            response = self.session.post(
                f"{self.base_url}/cctv/events/batch",
                json={"events": events},
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            logger.info(f"Batch sent: {result.get('successCount', 0)} success, "
                       f"{result.get('failCount', 0)} failed")

            if self._on_send_success:
                self._on_send_success(result)

            return True

        except Exception as e:
            logger.error(f"Batch send failed: {e}")
            if self._on_send_error:
                self._on_send_error(e, events)
            return False

    def _send_single(self, event_dict: Dict) -> bool:
        """Send a single event to the API."""
        try:
            response = self.session.post(
                f"{self.base_url}/cctv/event",
                json=event_dict,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            logger.debug(f"Event sent: {result.get('event_id')}")

            if self._on_send_success:
                self._on_send_success(result)

            return True

        except Exception as e:
            logger.error(f"Single send failed: {e}")
            if self._on_send_error:
                self._on_send_error(e, event_dict)
            return False

    def send_event(
        self,
        event_type: str,
        identity: Optional[str] = None,
        identity_confidence: Optional[float] = None,
        box_count: int = 0,
        direction: Optional[str] = None,
        track_id: Optional[int] = None,
        details: Optional[Dict] = None,
        image: Optional[bytes] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Send a CCTV event.

        Args:
            event_type: 'entered' or 'exited'
            identity: Identified person name
            identity_confidence: Confidence score (0-1)
            box_count: Number of boxes detected
            direction: 'entering' or 'exiting'
            track_id: Tracking ID
            details: Additional details dict
            image: Image bytes (will be base64 encoded)
            timestamp: Event timestamp (defaults to now)

        Returns:
            True if event was queued/sent successfully
        """
        if timestamp is None:
            timestamp = datetime.now()

        event = CCTVEvent(
            timestamp=timestamp.isoformat(),
            event_type=event_type,
            track_id=track_id,
            identity=identity,
            identity_confidence=identity_confidence,
            box_count=box_count,
            direction=direction,
            details=details,
            camera_id=self.camera_id,
            image_base64=base64.b64encode(image).decode('utf-8') if image else None
        )

        event_dict = event.to_dict()

        if self.async_mode:
            self._event_queue.put(event_dict)
            return True
        else:
            return self._send_single(event_dict)

    def send_entry(
        self,
        identity: Optional[str] = None,
        identity_confidence: Optional[float] = None,
        box_count: int = 0,
        track_id: Optional[int] = None,
        image: Optional[bytes] = None,
        **kwargs
    ) -> bool:
        """Convenience method for entry events."""
        return self.send_event(
            event_type='entered',
            identity=identity,
            identity_confidence=identity_confidence,
            box_count=box_count,
            direction='entering',
            track_id=track_id,
            image=image,
            **kwargs
        )

    def send_exit(
        self,
        identity: Optional[str] = None,
        identity_confidence: Optional[float] = None,
        box_count: int = 0,
        track_id: Optional[int] = None,
        image: Optional[bytes] = None,
        **kwargs
    ) -> bool:
        """Convenience method for exit events."""
        return self.send_event(
            event_type='exited',
            identity=identity,
            identity_confidence=identity_confidence,
            box_count=box_count,
            direction='exiting',
            track_id=track_id,
            image=image,
            **kwargs
        )

    def health_check(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = self.session.get(
                f"{self.base_url}/cctv/health",
                verify=self.verify_ssl,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def on_success(self, callback: Callable):
        """Register callback for successful sends."""
        self._on_send_success = callback

    def on_error(self, callback: Callable):
        """Register callback for failed sends."""
        self._on_send_error = callback

    def flush(self):
        """Force send all pending events."""
        if self._pending_batch:
            self._send_batch(self._pending_batch.copy())
            self._pending_batch.clear()
            self._last_batch_time = time.time()

    def stop(self):
        """Stop the background worker and send remaining events."""
        if self._worker_thread:
            self._stop_event.set()
            self._worker_thread.join(timeout=10)
            logger.info("CCTV API client stopped")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Singleton instance for easy access
_client: Optional[CCTVAPIClient] = None


def get_client() -> Optional[CCTVAPIClient]:
    """Get the global CCTV API client instance."""
    return _client


def init_client(**kwargs) -> CCTVAPIClient:
    """Initialize the global CCTV API client."""
    global _client
    _client = CCTVAPIClient(**kwargs)
    return _client


def send_event(*args, **kwargs) -> bool:
    """Send event using global client."""
    if _client:
        return _client.send_event(*args, **kwargs)
    return False
