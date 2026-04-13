"""
SQLite Database for Inventory Tracking
Stores shelf counts, changes, and history.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class ShelfCount:
    """Current shelf inventory count."""
    camera: str
    shelf: str
    count: int
    timestamp: datetime


@dataclass
class InventoryChange:
    """Record of inventory change."""
    id: int
    camera: str
    shelf: str
    old_count: int
    new_count: int
    change: int
    timestamp: datetime


class InventoryDatabase:
    """SQLite database for inventory tracking."""

    def __init__(self, db_path: str = "rack_data/inventory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Current inventory state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS current_inventory (
                camera TEXT,
                shelf TEXT,
                count INTEGER,
                last_updated TEXT,
                PRIMARY KEY (camera, shelf)
            )
        """)

        # Inventory history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera TEXT,
                shelf TEXT,
                count INTEGER,
                timestamp TEXT
            )
        """)

        # Change log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera TEXT,
                shelf TEXT,
                old_count INTEGER,
                new_count INTEGER,
                change INTEGER,
                timestamp TEXT
            )
        """)

        # Alerts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera TEXT,
                shelf TEXT,
                alert_type TEXT,
                message TEXT,
                acknowledged INTEGER DEFAULT 0,
                timestamp TEXT
            )
        """)

        conn.commit()
        conn.close()

    def update_count(self, camera: str, shelf: str, count: int) -> Optional[int]:
        """
        Update shelf count. Returns change amount if changed, None if same.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        # Get current count
        cursor.execute(
            "SELECT count FROM current_inventory WHERE camera=? AND shelf=?",
            (camera, shelf)
        )
        row = cursor.fetchone()
        old_count = row[0] if row else 0

        # Update current
        cursor.execute("""
            INSERT OR REPLACE INTO current_inventory (camera, shelf, count, last_updated)
            VALUES (?, ?, ?, ?)
        """, (camera, shelf, count, now))

        # Add to history
        cursor.execute("""
            INSERT INTO inventory_history (camera, shelf, count, timestamp)
            VALUES (?, ?, ?, ?)
        """, (camera, shelf, count, now))

        # Record change if different
        change = count - old_count
        if change != 0:
            cursor.execute("""
                INSERT INTO changes (camera, shelf, old_count, new_count, change, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (camera, shelf, old_count, count, change, now))

            # Create alert for significant changes
            if abs(change) >= 3:
                alert_type = "large_decrease" if change < 0 else "large_increase"
                message = f"{shelf}: {old_count} → {count} ({change:+d})"
                cursor.execute("""
                    INSERT INTO alerts (camera, shelf, alert_type, message, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (camera, shelf, alert_type, message, now))

        conn.commit()
        conn.close()

        return change if change != 0 else None

    def get_current_inventory(self, camera: str = None) -> List[ShelfCount]:
        """Get current inventory counts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if camera:
            cursor.execute(
                "SELECT camera, shelf, count, last_updated FROM current_inventory WHERE camera=?",
                (camera,)
            )
        else:
            cursor.execute("SELECT camera, shelf, count, last_updated FROM current_inventory")

        results = []
        for row in cursor.fetchall():
            results.append(ShelfCount(
                camera=row[0],
                shelf=row[1],
                count=row[2],
                timestamp=datetime.fromisoformat(row[3])
            ))

        conn.close()
        return results

    def get_recent_changes(self, limit: int = 20, camera: str = None) -> List[InventoryChange]:
        """Get recent inventory changes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if camera:
            cursor.execute("""
                SELECT id, camera, shelf, old_count, new_count, change, timestamp
                FROM changes WHERE camera=? ORDER BY timestamp DESC LIMIT ?
            """, (camera, limit))
        else:
            cursor.execute("""
                SELECT id, camera, shelf, old_count, new_count, change, timestamp
                FROM changes ORDER BY timestamp DESC LIMIT ?
            """, (limit,))

        results = []
        for row in cursor.fetchall():
            results.append(InventoryChange(
                id=row[0],
                camera=row[1],
                shelf=row[2],
                old_count=row[3],
                new_count=row[4],
                change=row[5],
                timestamp=datetime.fromisoformat(row[6])
            ))

        conn.close()
        return results

    def get_history(self, camera: str, shelf: str, hours: int = 24) -> List[Tuple[datetime, int]]:
        """Get count history for a shelf."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since = (datetime.now() - timedelta(hours=hours)).isoformat()

        cursor.execute("""
            SELECT timestamp, count FROM inventory_history
            WHERE camera=? AND shelf=? AND timestamp > ?
            ORDER BY timestamp
        """, (camera, shelf, since))

        results = [(datetime.fromisoformat(row[0]), row[1]) for row in cursor.fetchall()]

        conn.close()
        return results

    def get_unacknowledged_alerts(self) -> List[Dict]:
        """Get alerts that haven't been acknowledged."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, camera, shelf, alert_type, message, timestamp
            FROM alerts WHERE acknowledged=0 ORDER BY timestamp DESC
        """)

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "camera": row[1],
                "shelf": row[2],
                "type": row[3],
                "message": row[4],
                "timestamp": row[5],
            })

        conn.close()
        return results

    def acknowledge_alert(self, alert_id: int):
        """Mark alert as acknowledged."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
        conn.commit()
        conn.close()

    def get_summary(self) -> Dict:
        """Get inventory summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total items
        cursor.execute("SELECT SUM(count) FROM current_inventory")
        total = cursor.fetchone()[0] or 0

        # By camera
        cursor.execute("""
            SELECT camera, SUM(count) FROM current_inventory GROUP BY camera
        """)
        by_camera = {row[0]: row[1] for row in cursor.fetchall()}

        # Changes today
        today = datetime.now().replace(hour=0, minute=0, second=0).isoformat()
        cursor.execute("""
            SELECT COUNT(*) FROM changes WHERE timestamp > ?
        """, (today,))
        changes_today = cursor.fetchone()[0]

        # Pending alerts
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged=0")
        pending_alerts = cursor.fetchone()[0]

        conn.close()

        return {
            "total_items": total,
            "by_camera": by_camera,
            "changes_today": changes_today,
            "pending_alerts": pending_alerts,
        }
