#!/usr/bin/env python3
"""
BOX TRACKER WITH QR CODE DETECTION
Uses YOLOv8 for box detection + OpenCV for QR code reading.
QR codes on boxes are automatically linked to the box they're on.

Usage:
    python count_scan_place_qr.py <camera_ip>
    python count_scan_place_qr.py <camera_ip> --headless
"""

import cv2
import numpy as np
import os
import sys
import time
import threading
import queue
from datetime import datetime
from collections import deque

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ============ CONFIGURATION ============
CONFIG = {
    # Camera
    "rtsp_user": "fasspay",
    "rtsp_pass": "fasspay2025",
    "use_hd": True,

    # YOLOv8 Detection
    "model_path": "/home/jetson/inventory_security/runs/detect/train6/weights/best.pt",
    "confidence": 0.15,
    "detect_interval": 0.3,
    "input_size": 480,

    # Box filtering
    "min_area": 500,
    "max_area": 500000,

    # Tracking
    "iou_threshold": 0.15,
    "max_move_distance": 150,
    "ghost_timeout": 60,          # Seconds to remember removed boxes

    # QR Detection
    "qr_link_distance": 100,      # Max pixels between QR center and box center to link

    # Display
    "display_width": 1280,
    "display_height": 720,
    "headless": False,

    # Logging
    "log_file": "qr_tracking_log.csv",
}
# =======================================


class FrameGrabber:
    def __init__(self, url):
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        gst = (
            f"rtspsrc location={url} latency=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )
        self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return (True, self.frame.copy()) if self.frame is not None else (False, None)

    def release(self):
        self.running = False
        self.cap.release()


class QRBoxTracker:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.qr_detector = cv2.QRCodeDetector()

        self.confidence = config["confidence"]
        self.min_area = config["min_area"]
        self.max_area = config["max_area"]

        # Active boxes (currently visible)
        self.tracked_boxes = []
        self.next_box_id = 1

        # Ghost boxes (recently removed, remembered)
        self.ghost_boxes = []
        self.ghost_timeout = config["ghost_timeout"]

        # Inventory
        self.inventory = {}  # box_id -> {sku, zone, time, status}

        # Stats
        self.total_checked_in = 0
        self.total_checked_out = 0

    def load(self):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Device: {self.device}")

            model_path = self.config["model_path"]
            print(f"  Loading YOLOv8: {model_path}")
            self.model = YOLO(model_path)

            print("  Warming up...")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)

            print("  QR detector ready")
            print("  System ready!")
            return True

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def detect_qr_codes(self, frame):
        """Detect QR codes in frame."""
        qr_codes = []

        # Try OpenCV QR detector
        try:
            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(frame)
            if retval and points is not None:
                for i, data in enumerate(decoded_info):
                    if data:  # Non-empty QR data
                        pts = points[i].astype(int)
                        # Calculate center
                        cx = int(np.mean(pts[:, 0]))
                        cy = int(np.mean(pts[:, 1]))
                        qr_codes.append({
                            'data': data,
                            'center': (cx, cy),
                            'points': pts
                        })
        except:
            pass

        # Fallback: try single QR detection
        if not qr_codes:
            try:
                data, points, _ = self.qr_detector.detectAndDecode(frame)
                if data and points is not None:
                    pts = points[0].astype(int)
                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))
                    qr_codes.append({
                        'data': data,
                        'center': (cx, cy),
                        'points': pts
                    })
            except:
                pass

        return qr_codes

    def detect_boxes(self, frame):
        """Detect boxes using YOLOv8."""
        if self.model is None:
            return []

        h, w = frame.shape[:2]

        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.config["input_size"],
            verbose=False
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])

                bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
                area = (bx2 - bx1) * (by2 - by1)

                if self.min_area <= area <= self.max_area:
                    detections.append({
                        'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2,
                        'cx': (bx1 + bx2) // 2,
                        'cy': (by1 + by2) // 2,
                        'score': score
                    })

        return detections

    def link_qr_to_boxes(self, detections, qr_codes):
        """Link QR codes to nearby boxes."""
        max_dist = self.config["qr_link_distance"]

        for qr in qr_codes:
            qx, qy = qr['center']
            best_box = None
            best_dist = float('inf')

            for det in detections:
                # Check if QR center is inside box or nearby
                if det['x1'] <= qx <= det['x2'] and det['y1'] <= qy <= det['y2']:
                    # QR is inside box - perfect match
                    det['qr_data'] = qr['data']
                    det['qr_points'] = qr['points']
                    best_box = None  # Already assigned
                    break
                else:
                    # Calculate distance to box center
                    dist = ((qx - det['cx'])**2 + (qy - det['cy'])**2) ** 0.5
                    if dist < best_dist and dist < max_dist:
                        best_dist = dist
                        best_box = det

            if best_box is not None and 'qr_data' not in best_box:
                best_box['qr_data'] = qr['data']
                best_box['qr_points'] = qr['points']

        return detections

    def _iou(self, box1, box2):
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        return inter / (area1 + area2 - inter + 1e-6)

    def _distance(self, box1, box2):
        return ((box1['cx'] - box2['cx'])**2 + (box1['cy'] - box2['cy'])**2) ** 0.5

    def track_boxes(self, detections):
        """Track boxes with ghost memory."""
        now = time.time()

        # Mark all tracked boxes as missing
        for tb in self.tracked_boxes:
            tb['missing'] += 1

        matched_tracks = set()
        matched_dets = set()

        # First pass: match by QR code (most reliable)
        for i, det in enumerate(detections):
            if 'qr_data' not in det:
                continue

            qr = det['qr_data']

            # Check active boxes
            for j, tb in enumerate(self.tracked_boxes):
                if j in matched_tracks:
                    continue
                if tb.get('qr_data') == qr:
                    # Same QR - same box!
                    tb.update({
                        'x1': det['x1'], 'y1': det['y1'],
                        'x2': det['x2'], 'y2': det['y2'],
                        'cx': det['cx'], 'cy': det['cy'],
                        'missing': 0
                    })
                    det['box_id'] = tb['id']
                    det['sku'] = tb['qr_data']
                    det['is_new'] = False
                    matched_tracks.add(j)
                    matched_dets.add(i)
                    break

            # Check ghost boxes (returning box)
            if i not in matched_dets:
                for g, ghost in enumerate(self.ghost_boxes):
                    if ghost.get('qr_data') == qr:
                        # Box returned! Resurrect from ghost
                        print(f"\n  [RETURNED] Box #{ghost['id']} ({qr}) is back!")

                        new_box = {
                            'id': ghost['id'],
                            'x1': det['x1'], 'y1': det['y1'],
                            'x2': det['x2'], 'y2': det['y2'],
                            'cx': det['cx'], 'cy': det['cy'],
                            'qr_data': qr,
                            'missing': 0,
                            'created': ghost['created']
                        }
                        self.tracked_boxes.append(new_box)

                        det['box_id'] = ghost['id']
                        det['sku'] = qr
                        det['is_new'] = False
                        matched_dets.add(i)

                        # Update inventory
                        if ghost['id'] in self.inventory:
                            self.inventory[ghost['id']]['status'] = 'active'
                            self.inventory[ghost['id']]['returned'] = datetime.now().isoformat()

                        self.ghost_boxes.pop(g)
                        break

        # Second pass: match by position (for boxes without QR)
        for i, det in enumerate(detections):
            if i in matched_dets:
                continue

            best_score = 0
            best_idx = None

            for j, tb in enumerate(self.tracked_boxes):
                if j in matched_tracks:
                    continue

                iou = self._iou(det, tb)
                if iou > 0.3:
                    score = iou + 1.0
                    if score > best_score:
                        best_score = score
                        best_idx = j
                else:
                    dist = self._distance(det, tb)
                    if dist < self.config["max_move_distance"]:
                        score = 1.0 - (dist / self.config["max_move_distance"])
                        if score > best_score:
                            best_score = score
                            best_idx = j

            if best_idx is not None:
                tb = self.tracked_boxes[best_idx]
                tb.update({
                    'x1': det['x1'], 'y1': det['y1'],
                    'x2': det['x2'], 'y2': det['y2'],
                    'cx': det['cx'], 'cy': det['cy'],
                    'missing': 0
                })

                # Update QR if newly detected
                if 'qr_data' in det and 'qr_data' not in tb:
                    tb['qr_data'] = det['qr_data']
                    print(f"\n  [LINKED] Box #{tb['id']} = {det['qr_data']}")

                    # Add to inventory
                    self.inventory[tb['id']] = {
                        'sku': det['qr_data'],
                        'zone': self._get_zone(det),
                        'time': datetime.now().isoformat(),
                        'status': 'active'
                    }
                    self.total_checked_in += 1

                det['box_id'] = tb['id']
                det['sku'] = tb.get('qr_data')
                det['is_new'] = False
                matched_tracks.add(best_idx)
                matched_dets.add(i)

        # Third pass: create new boxes
        for i, det in enumerate(detections):
            if i in matched_dets:
                continue

            new_box = {
                'id': self.next_box_id,
                'x1': det['x1'], 'y1': det['y1'],
                'x2': det['x2'], 'y2': det['y2'],
                'cx': det['cx'], 'cy': det['cy'],
                'missing': 0,
                'created': now
            }

            if 'qr_data' in det:
                new_box['qr_data'] = det['qr_data']
                print(f"\n  [NEW] Box #{self.next_box_id} = {det['qr_data']}")

                self.inventory[self.next_box_id] = {
                    'sku': det['qr_data'],
                    'zone': self._get_zone(det),
                    'time': datetime.now().isoformat(),
                    'status': 'active'
                }
                self.total_checked_in += 1
            else:
                print(f"\n  [NEW] Box #{self.next_box_id} (no QR)")

            self.tracked_boxes.append(new_box)
            det['box_id'] = self.next_box_id
            det['sku'] = new_box.get('qr_data')
            det['is_new'] = True
            self.next_box_id += 1

        # Move missing boxes to ghosts
        still_active = []
        for tb in self.tracked_boxes:
            if tb['missing'] > 5:  # ~5 frames missing
                if tb.get('qr_data'):
                    # Has QR - remember as ghost
                    tb['ghost_time'] = now
                    self.ghost_boxes.append(tb)
                    print(f"\n  [REMOVED] Box #{tb['id']} ({tb['qr_data']}) - watching for return...")

                    if tb['id'] in self.inventory:
                        self.inventory[tb['id']]['status'] = 'removed'
                        self.inventory[tb['id']]['removed_time'] = datetime.now().isoformat()
            else:
                still_active.append(tb)

        self.tracked_boxes = still_active

        # Expire old ghosts
        active_ghosts = []
        for ghost in self.ghost_boxes:
            if now - ghost['ghost_time'] > self.ghost_timeout:
                print(f"\n  [CHECKED OUT] Box #{ghost['id']} ({ghost['qr_data']})")
                if ghost['id'] in self.inventory:
                    self.inventory[ghost['id']]['status'] = 'checked_out'
                    self.inventory[ghost['id']]['checkout_time'] = datetime.now().isoformat()
                self.total_checked_out += 1
            else:
                active_ghosts.append(ghost)

        self.ghost_boxes = active_ghosts

        return detections

    def _get_zone(self, det):
        cx, cy = det['cx'], det['cy']
        col = chr(ord('A') + (cx // 200) % 4)
        row = str((cy // 200) % 4 + 1)
        return f"{col}{row}"

    def draw(self, frame, detections, qr_codes):
        """Draw boxes, QR codes, and status."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw QR codes
        for qr in qr_codes:
            pts = qr['points']
            cv2.polylines(display, [pts], True, (255, 0, 255), 2)
            cv2.putText(display, qr['data'][:20], (pts[0][0], pts[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Draw boxes
        for det in detections:
            if det.get('sku'):
                color = (0, 255, 0)  # Green - has QR
                label = det['sku'][:16]
            else:
                color = (0, 165, 255)  # Orange - no QR yet
                label = f"#{det.get('box_id', '?')}"

            thickness = 3 if det.get('is_new') else 2
            cv2.rectangle(display, (det['x1'], det['y1']),
                         (det['x2'], det['y2']), color, thickness)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (det['x1'], det['y1'] - th - 8),
                         (det['x1'] + tw + 8, det['y1']), color, -1)
            cv2.putText(display, label, (det['x1'] + 4, det['y1'] - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Info panel
        cv2.rectangle(display, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.putText(display, f"Boxes: {len(detections)}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        tagged = sum(1 for d in detections if d.get('sku'))
        cv2.putText(display, f"Tagged: {tagged}  Untagged: {len(detections) - tagged}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(display, f"Ghosts: {len(self.ghost_boxes)}  Checked out: {self.total_checked_out}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(display, f"QR detected: {len(qr_codes)}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Help
        cv2.rectangle(display, (0, h - 30), (w, h), (30, 30, 30), -1)
        cv2.putText(display, "Q:Quit | I:Inventory | +/-:Threshold | R:ROI",
                   (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return display

    def print_inventory(self):
        """Print inventory status."""
        print("\n" + "=" * 60)
        print("  INVENTORY STATUS")
        print("=" * 60)

        active = [(k, v) for k, v in self.inventory.items() if v['status'] == 'active']
        removed = [(k, v) for k, v in self.inventory.items() if v['status'] == 'removed']
        checked_out = [(k, v) for k, v in self.inventory.items() if v['status'] == 'checked_out']

        if active:
            print(f"\n  ACTIVE ({len(active)}):")
            for box_id, info in active:
                print(f"    Box #{box_id}: {info['sku']} @ {info['zone']}")

        if removed:
            print(f"\n  TEMPORARILY REMOVED ({len(removed)}):")
            for box_id, info in removed:
                print(f"    Box #{box_id}: {info['sku']} (watching...)")

        if checked_out:
            print(f"\n  CHECKED OUT ({len(checked_out)}):")
            for box_id, info in checked_out:
                print(f"    Box #{box_id}: {info['sku']} @ {info.get('checkout_time', 'unknown')}")

        if not self.inventory:
            print("  No items tracked yet")

        print("=" * 60 + "\n")


def run(camera_ip):
    print("=" * 50)
    print("  QR BOX TRACKER")
    print("  YOLOv8 + QR Detection")
    print("=" * 50)

    # Connect camera
    stream = "stream1" if CONFIG["use_hd"] else "stream2"
    url = f"rtsp://{CONFIG['rtsp_user']}:{CONFIG['rtsp_pass']}@{camera_ip}:554/{stream}"

    print(f"\nConnecting...")
    tracker = QRBoxTracker(CONFIG)
    if not tracker.load():
        return

    cap = FrameGrabber(url)
    time.sleep(1)

    ret, frame = cap.read()
    if not ret:
        print("Connection failed!")
        return

    print(f"Connected! {frame.shape[1]}x{frame.shape[0]}")

    headless = CONFIG["headless"]
    if not headless:
        try:
            cv2.namedWindow("QR Box Tracker", cv2.WINDOW_NORMAL)
        except:
            headless = True

    # Logging
    log_file = CONFIG.get("log_file")
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"\n# Session: {datetime.now().isoformat()}\n")
            f.write("timestamp,boxes,tagged,qr_codes,ghosts,checked_out\n")

    last_detect = 0
    detect_interval = CONFIG["detect_interval"]
    detections = []
    qr_codes = []

    print("\n>>> Place boxes with QR stickers in view <<<\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Run detection
            if time.time() - last_detect >= detect_interval:
                last_detect = time.time()

                start = time.time()

                # Detect QR codes
                qr_codes = tracker.detect_qr_codes(frame)

                # Detect boxes
                box_dets = tracker.detect_boxes(frame)

                # Link QR to boxes
                box_dets = tracker.link_qr_to_boxes(box_dets, qr_codes)

                # Track
                detections = tracker.track_boxes(box_dets)

                elapsed = time.time() - start
                tagged = sum(1 for d in detections if d.get('sku'))

                print(f"  Detection: {elapsed:.3f}s | Boxes: {len(detections)} | QR: {len(qr_codes)} | Tagged: {tagged}")

                if log_file:
                    with open(log_file, "a") as f:
                        f.write(f"{datetime.now().isoformat()},{len(detections)},{tagged},{len(qr_codes)},{len(tracker.ghost_boxes)},{tracker.total_checked_out}\n")

            # Display
            if not headless:
                display = tracker.draw(frame, detections, qr_codes)

                dh, dw = display.shape[:2]
                scale = min(CONFIG["display_width"] / dw, CONFIG["display_height"] / dh, 1.0)
                if scale < 1:
                    display = cv2.resize(display, (int(dw * scale), int(dh * scale)))

                cv2.imshow("QR Box Tracker", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('i'):
                    tracker.print_inventory()
                elif key in (ord('+'), ord('=')):
                    tracker.confidence = min(0.9, tracker.confidence + 0.05)
                    print(f"  Threshold: {tracker.confidence:.2f}")
                elif key in (ord('-'), ord('_')):
                    tracker.confidence = max(0.05, tracker.confidence - 0.05)
                    print(f"  Threshold: {tracker.confidence:.2f}")
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.print_inventory()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_scan_place_qr.py <camera_ip> [--headless]")
        print("\nThis script:")
        print("  - Detects boxes using YOLOv8")
        print("  - Reads QR codes on boxes automatically")
        print("  - Links QR code to the box it's on")
        print("  - Tracks boxes even when moved")
        print("  - Detects checkout (box removed for 60s)")
    else:
        if "--headless" in sys.argv:
            CONFIG["headless"] = True
        run(sys.argv[1])
