"""
API Client for sending inventory scan events to ops-portal.

Sends box tracking events (new, removed, returned, checked_out) to
the ops-portal inventory service.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from queue import Queue, Empty
from typing import Optional, Dict, Any, List, Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class InventoryEvent:
    """Data class representing an inventory scan event."""
    timestamp: str
    event_type: str  # 'box_new', 'box_removed', 'box_returned', 'box_checked_out'
    sku: str
    camera_id: Optional[str] = None
    box_id: Optional[int] = None
    zone: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


class InventoryAPIClient:
    """
    Client for sending inventory events to the ops-portal API.

    Features:
    - Automatic retry with exponential backoff
    - Background queue for async sending
    - Batch upload support
    - Connection pooling
    - SSL verification configurable
    """

    def __init__(
        self,
        base_url: str = "https://ops-portal.fasspay.com/report/inventory",
        camera_id: str = "jetson-scanner-01",
        verify_ssl: bool = True,
        timeout: int = 30,
        max_retries: int = 3,
        batch_size: int = 10,
        batch_interval: float = 5.0,
        async_mode: bool = True
    ):
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

        # Heartbeat
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_interval: int = 30

        if async_mode:
            self._start_worker()

    def _start_worker(self):
        """Start background worker thread for async sending."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Inventory API client worker started")

    def _worker_loop(self):
        """Background worker loop for processing queued events."""
        while not self._stop_event.is_set():
            try:
                try:
                    event_dict = self._event_queue.get(timeout=1.0)
                    self._pending_batch.append(event_dict)
                except Empty:
                    pass

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
                f"{self.base_url}/events/batch",
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
                f"{self.base_url}/event",
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
        sku: str,
        box_id: Optional[int] = None,
        zone: Optional[str] = None,
        details: Optional[Dict] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Send an inventory event.

        Args:
            event_type: 'box_new', 'box_removed', 'box_returned', 'box_checked_out'
            sku: SKU/QR code data
            box_id: Local tracking box ID
            zone: Zone identifier (e.g., 'A1')
            details: Additional details dict
            timestamp: Event timestamp (defaults to now)

        Returns:
            True if event was queued/sent successfully
        """
        if timestamp is None:
            timestamp = datetime.now()

        event = InventoryEvent(
            timestamp=timestamp.isoformat(),
            event_type=event_type,
            sku=sku,
            camera_id=self.camera_id,
            box_id=box_id,
            zone=zone,
            details=details
        )

        event_dict = event.to_dict()

        if self.async_mode:
            self._event_queue.put(event_dict)
            return True
        else:
            return self._send_single(event_dict)

    def send_box_new(self, sku: str, box_id: int = None, zone: str = None, **kwargs) -> bool:
        """Send event when new box with QR is detected."""
        return self.send_event('box_new', sku, box_id=box_id, zone=zone, **kwargs)

    def send_box_removed(self, sku: str, box_id: int = None, **kwargs) -> bool:
        """Send event when box is temporarily removed from view."""
        return self.send_event('box_removed', sku, box_id=box_id, **kwargs)

    def send_box_returned(self, sku: str, box_id: int = None, **kwargs) -> bool:
        """Send event when previously removed box returns."""
        return self.send_event('box_returned', sku, box_id=box_id, **kwargs)

    def send_box_checked_out(self, sku: str, box_id: int = None, **kwargs) -> bool:
        """Send event when box is confirmed checked out (gone > timeout)."""
        return self.send_event('box_checked_out', sku, box_id=box_id, **kwargs)

    def send_heartbeat(self, status: str = "running") -> bool:
        """Send heartbeat to indicate this scanner is online."""
        try:
            response = self.session.post(
                f"{self.base_url}/heartbeat",
                json={"camera_id": self.camera_id, "status": status},
                verify=self.verify_ssl,
                timeout=10
            )
            response.raise_for_status()
            logger.debug(f"Heartbeat sent: {self.camera_id}")
            return True
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            return False

    def start_heartbeat(self, interval: int = 30):
        """Start background heartbeat thread."""
        self._heartbeat_interval = interval
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info(f"Heartbeat started (every {interval}s)")

    def _heartbeat_loop(self):
        """Background loop for sending heartbeats."""
        while not self._stop_event.is_set():
            self.send_heartbeat("running")
            self._stop_event.wait(self._heartbeat_interval)
        self.send_heartbeat("stopped")

    def health_check(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
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
        """Stop the background worker and heartbeat."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        logger.info("Inventory API client stopped")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Singleton instance
_client: Optional[InventoryAPIClient] = None


def get_client() -> Optional[InventoryAPIClient]:
    """Get the global inventory API client instance."""
    return _client


def init_client(**kwargs) -> InventoryAPIClient:
    """Initialize the global inventory API client."""
    global _client
    _client = InventoryAPIClient(**kwargs)
    return _client


def send_event(*args, **kwargs) -> bool:
    """Send event using global client."""
    if _client:
        return _client.send_event(*args, **kwargs)
    return False
