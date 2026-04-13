#!/usr/bin/env python3
"""
SCAN-THEN-PLACE Inventory System
1. Scan barcode with handheld scanner
2. Place box on rack
3. AI auto-assigns SKU to new box

No QR stickers needed!
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

# Add GroundingDINO to path
GDINO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GroundingDINO")
sys.path.insert(0, GDINO_PATH)


class ScanPlaceTracker:
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.roi = None

        # ============ ADJUST THESE ============
        self.confidence = 0.20          # Detection threshold (lower = more boxes)
        self.prompt = "cardboard box . brown box . package"  # Detection prompt
        self.min_area = 500             # Min box size in pixels
        self.max_area = 500000          # Max box size in pixels
        # ======================================

        # Count stabilization (median filter)
        self.count_history = []
        self.stable_count = 0

        # BOX TRACKING
        self.tracked_boxes = []
        self.next_box_id = 1
        self.iou_threshold = 0.15  # For overlapping boxes
        self.max_move_distance = 150  # Max pixels a box can move between frames

        # Assignment feedback
        self.last_assignment = None  # (sku, time) for showing confirmation

        # SCAN-THEN-PLACE QUEUE
        self.pending_skus = deque()  # SKUs waiting to be assigned
        self.sku_timeout = 60  # Seconds before pending SKU expires (increased)

        # INVENTORY
        self.inventory = {}  # box_id -> {sku, zone, time}
        self.zones = {}  # Define rack zones

        # Stats
        self.total_scanned = 0
        self.total_placed = 0

    def load(self):
        """Load detection model."""
        try:
            import torch
            from groundingdino.util.inference import load_model

            print("Loading GroundingDINO...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            config_path = os.path.join(GDINO_PATH, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
            weights_path = os.path.join(GDINO_PATH, "weights", "groundingdino_swint_ogc.pth")

            self.model = load_model(config_path, weights_path, device=self.device)
            print("  GroundingDINO ready!")
            return True

        except Exception as e:
            print(f"Error: {e}")
            return False

    def queue_sku(self, sku):
        """Add SKU to pending queue (from barcode scanner)."""
        self.pending_skus.append({
            'sku': sku,
            'time': time.time()
        })
        self.total_scanned += 1
        print(f"\n  [SCANNED] {sku} - now place box on rack!")
        print(f"  Pending SKUs: {len(self.pending_skus)}")

    def get_pending_sku(self):
        """Get next pending SKU (remove expired ones)."""
        now = time.time()

        # Remove expired SKUs
        while self.pending_skus:
            if now - self.pending_skus[0]['time'] > self.sku_timeout:
                expired = self.pending_skus.popleft()
                print(f"  [EXPIRED] {expired['sku']} - took too long!")
            else:
                break

        if self.pending_skus:
            return self.pending_skus[0]['sku']
        return None

    def get_pending_info(self):
        """Get pending SKU with time remaining."""
        now = time.time()

        # Clean expired first
        while self.pending_skus:
            if now - self.pending_skus[0]['time'] > self.sku_timeout:
                expired = self.pending_skus.popleft()
                print(f"  [EXPIRED] {expired['sku']} - took too long!")
            else:
                break

        if self.pending_skus:
            item = self.pending_skus[0]
            remaining = self.sku_timeout - (now - item['time'])
            return item['sku'], remaining
        return None, 0

    def consume_pending_sku(self):
        """Remove and return first pending SKU."""
        if self.pending_skus:
            item = self.pending_skus.popleft()
            return item['sku']
        return None

    def detect_boxes(self, frame):
        """Detect boxes (downscaled for speed)."""
        if self.model is None:
            return []

        import torch
        from PIL import Image
        import groundingdino.datasets.transforms as T
        from groundingdino.util.inference import predict

        h, w = frame.shape[:2]
        work = frame
        offset = (0, 0)

        if self.roi:
            x1, y1 = int(self.roi[0]*w), int(self.roi[1]*h)
            x2, y2 = int(self.roi[2]*w), int(self.roi[3]*h)
            work = frame[y1:y2, x1:x2]
            offset = (x1, y1)

        # Downscale for faster detection (larger = more accurate, smaller = faster)
        orig_work_h, orig_work_w = work.shape[:2]
        max_dim = 1000  # Increased from 640 for better accuracy
        detect_scale = min(max_dim / orig_work_w, max_dim / orig_work_h, 1.0)
        if detect_scale < 1.0:
            work = cv2.resize(work, (int(orig_work_w * detect_scale), int(orig_work_h * detect_scale)))

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        image_pil = Image.fromarray(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))
        image_tensor, _ = transform(image_pil, None)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=self.prompt,
            box_threshold=self.confidence,
            text_threshold=self.confidence,
            device=self.device
        )

        detections = []
        detect_h, detect_w = work.shape[:2]  # After downscale

        # Scale back to original ROI size
        if detect_scale < 1.0:
            scale_back = 1.0 / detect_scale
        else:
            scale_back = 1.0

        for box, score in zip(boxes, logits):
            cx, cy, bw, bh = box.tolist()
            bx1 = int((cx - bw/2) * detect_w * scale_back) + offset[0]
            by1 = int((cy - bh/2) * detect_h * scale_back) + offset[1]
            bx2 = int((cx + bw/2) * detect_w * scale_back) + offset[0]
            by2 = int((cy + bh/2) * detect_h * scale_back) + offset[1]

            area = (bx2 - bx1) * (by2 - by1)
            if self.min_area <= area <= self.max_area:
                detections.append({
                    'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2,
                    'score': float(score)
                })

        # NMS
        detections = self._nms(detections, 0.5)

        # Stabilize count (median filter)
        self.count_history.append(len(detections))
        if len(self.count_history) > 5:
            self.count_history.pop(0)
        self.stable_count = sorted(self.count_history)[len(self.count_history)//2]

        return detections

    def _nms(self, detections, threshold):
        if len(detections) <= 1:
            return detections

        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []

        for det in detections:
            overlap = False
            for kept in keep:
                if self._iou(det, kept) > threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(det)

        return keep

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

    def _centroid_dist(self, box1, box2):
        """Calculate distance between box centers."""
        cx1 = (box1['x1'] + box1['x2']) / 2
        cy1 = (box1['y1'] + box1['y2']) / 2
        cx2 = (box2['x1'] + box2['x2']) / 2
        cy2 = (box2['y1'] + box2['y2']) / 2
        return ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5

    def track_and_assign(self, detections):
        """Track boxes and assign pending SKUs to NEW boxes."""
        # Age tracked boxes
        for tb in self.tracked_boxes:
            tb['missing'] += 1

        matched_tracks = set()
        matched_dets = set()

        # Match detections to tracked boxes (IoU first, then centroid distance)
        for i, det in enumerate(detections):
            best_score = 0
            best_idx = None

            for j, tb in enumerate(self.tracked_boxes):
                if j in matched_tracks:
                    continue

                # Try IoU first
                iou = self._iou(det, tb)
                if iou >= self.iou_threshold:
                    score = iou + 1.0  # Prefer IoU matches
                    if score > best_score:
                        best_score = score
                        best_idx = j
                else:
                    # Fallback: centroid distance (for moving boxes)
                    dist = self._centroid_dist(det, tb)
                    if dist < self.max_move_distance:
                        score = 1.0 - (dist / self.max_move_distance)  # 0-1 score
                        if score > best_score:
                            best_score = score
                            best_idx = j

            if best_idx is not None:
                # Update existing tracked box
                tb = self.tracked_boxes[best_idx]
                tb['x1'], tb['y1'] = det['x1'], det['y1']
                tb['x2'], tb['y2'] = det['x2'], det['y2']
                tb['missing'] = 0

                det['box_id'] = tb['id']
                det['sku'] = tb['sku']
                det['is_new'] = False

                matched_tracks.add(best_idx)
                matched_dets.add(i)

        # NEW boxes - assign pending SKU!
        for i, det in enumerate(detections):
            if i not in matched_dets:
                # This is a NEW box!
                pending_sku = self.consume_pending_sku()

                new_box = {
                    'id': self.next_box_id,
                    'x1': det['x1'], 'y1': det['y1'],
                    'x2': det['x2'], 'y2': det['y2'],
                    'sku': pending_sku,
                    'missing': 0,
                    'created': time.time()
                }
                self.tracked_boxes.append(new_box)

                det['box_id'] = self.next_box_id
                det['sku'] = pending_sku
                det['is_new'] = True

                if pending_sku:
                    self.total_placed += 1
                    zone = self._get_zone(det)
                    self.inventory[self.next_box_id] = {
                        'sku': pending_sku,
                        'zone': zone,
                        'time': datetime.now().isoformat()
                    }
                    print(f"\n  *** [ASSIGNED] Box #{self.next_box_id} = {pending_sku} @ Zone {zone} ***")
                    self.last_assignment = (pending_sku, time.time())  # For visual feedback

                self.next_box_id += 1

        # Remove old boxes (faster removal = quicker detection of new boxes)
        self.tracked_boxes = [tb for tb in self.tracked_boxes if tb['missing'] <= 5]

        return detections

    def _get_zone(self, det):
        """Get zone based on box position (grid system)."""
        if not self.roi:
            return "A1"

        # Calculate relative position in ROI
        cx = (det['x1'] + det['x2']) / 2
        cy = (det['y1'] + det['y2']) / 2

        # Map to grid (e.g., A1, A2, B1, B2)
        col = int((cx - self.roi[0] * 1000) / 200) % 4  # A, B, C, D
        row = int((cy - self.roi[1] * 1000) / 200) % 4  # 1, 2, 3, 4

        zone_col = chr(ord('A') + col)
        zone_row = str(row + 1)

        return f"{zone_col}{zone_row}"

    def draw(self, frame, detections):
        """Draw boxes and inventory."""
        display = frame.copy()
        h, w = display.shape[:2]

        # ROI
        if self.roi:
            rx1, ry1 = int(self.roi[0]*w), int(self.roi[1]*h)
            rx2, ry2 = int(self.roi[2]*w), int(self.roi[3]*h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

        # Boxes
        assigned = 0
        unassigned = 0

        for det in detections:
            if det.get('sku'):
                color = (0, 255, 0)  # Green = has SKU
                assigned += 1

                # SKU label
                label = det['sku'][:16]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(display, (det['x1'], det['y1']-th-8),
                             (det['x1']+tw+8, det['y1']), (0, 180, 0), -1)
                cv2.putText(display, label, (det['x1']+4, det['y1']-4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            else:
                color = (0, 100, 255)  # Orange = no SKU
                unassigned += 1

                cv2.putText(display, f"#{det.get('box_id', '?')}", (det['x1']+5, det['y1']+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Highlight NEW boxes
            thickness = 4 if det.get('is_new') else 2
            cv2.rectangle(display, (det['x1'], det['y1']),
                         (det['x2'], det['y2']), color, thickness)

        # Info panel
        cv2.rectangle(display, (10, 10), (420, 140), (0, 0, 0), -1)

        cv2.putText(display, f"Boxes: {self.stable_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.putText(display, f"Tagged: {assigned}  Untagged: {unassigned}  Thresh: {self.confidence:.2f}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Pending SKU with countdown
        pending, remaining = self.get_pending_info()
        if pending:
            cv2.putText(display, f"NEXT: {pending}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"Place box now ({int(remaining)}s)", (20, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            # Flash warning if running low on time
            if remaining < 10:
                cv2.putText(display, "HURRY!", (20, 145),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(display, "Ready - scan SKU", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Show assignment confirmation (flash for 3 seconds)
        if self.last_assignment:
            sku, assign_time = self.last_assignment
            if time.time() - assign_time < 3.0:
                # Big green confirmation
                cv2.rectangle(display, (w//2 - 200, h//2 - 50), (w//2 + 200, h//2 + 50), (0, 180, 0), -1)
                cv2.putText(display, f"ASSIGNED: {sku}", (w//2 - 150, h//2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                self.last_assignment = None

        # Help
        cv2.rectangle(display, (0, h-30), (w, h), (30, 30, 30), -1)
        cv2.putText(display, "S:Sync/Live | +/-:Thresh | R:ROI | I:List | Q:Quit",
                   (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return display

    def set_roi(self, frame):
        """Draw ROI."""
        h, w = frame.shape[:2]
        scale = min(1280/w, 720/h, 1.0)
        display = cv2.resize(frame, (int(w*scale), int(h*scale))) if scale < 1 else frame.copy()

        pts, drawing, temp = [], False, display.copy()

        def mouse(event, x, y, flags, param):
            nonlocal pts, drawing, temp
            if event == cv2.EVENT_LBUTTONDOWN:
                pts, drawing = [(x, y)], True
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp = display.copy()
                cv2.rectangle(temp, pts[0], (x, y), (0, 255, 255), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                pts.append((x, y))
                drawing = False

        cv2.namedWindow("Draw ROI")
        cv2.setMouseCallback("Draw ROI", mouse)

        while True:
            cv2.imshow("Draw ROI", temp)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and len(pts) == 2:
                break
            elif key == 27:
                cv2.destroyWindow("Draw ROI")
                return

        cv2.destroyWindow("Draw ROI")
        dh, dw = display.shape[:2]
        self.roi = (min(pts[0][0], pts[1][0])/dw, min(pts[0][1], pts[1][1])/dh,
                   max(pts[0][0], pts[1][0])/dw, max(pts[0][1], pts[1][1])/dh)

    def print_inventory(self):
        """Print current inventory."""
        print("\n" + "=" * 50)
        print("  INVENTORY")
        print("=" * 50)
        if not self.inventory:
            print("  No items tracked yet")
        else:
            for box_id, info in self.inventory.items():
                print(f"  Box #{box_id}: {info['sku']} @ Zone {info['zone']}")
        print("=" * 50 + "\n")


class LatestFrameGrabber:
    """Grabs latest frame from RTSP, discarding old buffered frames."""

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()

    def _grab_loop(self):
        """Continuously grab frames, keeping only the latest."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        """Get the latest frame."""
        with self.lock:
            return self.frame is not None, self.frame.copy() if self.frame is not None else None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.cap.release()


def run(camera_ip):
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"

    # Use HD for accuracy, SD for speed
    use_hd = True  # Set False for faster but less accurate
    stream = "stream1" if use_hd else "stream2"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/{stream}"

    print("=" * 50)
    print("  SCAN-THEN-PLACE INVENTORY")
    print("=" * 50)
    print("""
  WORKFLOW:
  1. Scan barcode OR type SKU in terminal + Enter
  2. Place box on rack
  3. AI auto-assigns SKU to new box!

  Press Q in video window to quit.
    """)

    tracker = ScanPlaceTracker()
    if not tracker.load():
        return

    print(f"Connecting {'HD' if use_hd else 'SD'} stream (with live frame grabber)...")
    cap = LatestFrameGrabber(url)
    time.sleep(1)  # Wait for first frame

    if not cap.isOpened():
        print("Connection failed!")
        return

    print("Connected - LIVE feed (no delay)!")

    ret, frame = cap.read()
    if ret:
        print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
        tracker.set_roi(frame)

    cv2.namedWindow("Scan-Place Tracker", cv2.WINDOW_NORMAL)

    # Threading for detection
    detections = []
    detecting = False
    detect_frame = None
    detect_result = None
    display_frame = None  # Frame to show (the one that was detected)

    def detect_thread():
        nonlocal detect_result, detecting, display_frame
        if detect_frame is not None:
            try:
                dets = tracker.detect_boxes(detect_frame)
                dets = tracker.track_and_assign(dets)
                detect_result = dets
                display_frame = detect_frame.copy()  # Show THIS frame with boxes
            except Exception as e:
                print(f"Detection error: {e}")
        detecting = False

    last_detect = 0
    detect_interval = 1.0  # Balance speed vs accuracy (was 0.3)

    # Barcode scanner input via separate thread (reads from stdin)
    sku_queue = queue.Queue()
    running = True

    def stdin_reader():
        """Read barcode scanner input from stdin in separate thread."""
        while running:
            try:
                line = input()  # Blocks until Enter is pressed
                sku = line.strip().upper()
                if sku:
                    sku_queue.put(sku)
            except EOFError:
                break
            except Exception as e:
                if running:
                    print(f"Input error: {e}")

    # Start stdin reader thread
    input_thread = threading.Thread(target=stdin_reader, daemon=True)
    input_thread.start()
    print("\n>>> Ready! Scan barcode or type SKU here and press Enter <<<\n")

    # For manual typing in OpenCV window
    cv_sku_buffer = ""

    # View mode: True = synced (boxes match frame), False = live (smooth but boxes may lag)
    synced_view = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Check for barcode scanner input (from stdin thread)
            try:
                while True:
                    sku = sku_queue.get_nowait()
                    tracker.queue_sku(sku)
            except queue.Empty:
                pass

            # Background detection
            if not detecting and time.time() - last_detect >= detect_interval:
                detect_frame = frame.copy()
                detecting = True
                last_detect = time.time()
                threading.Thread(target=detect_thread, daemon=True).start()

            if detect_result is not None:
                detections = detect_result
                detect_result = None

            # Synced = boxes match frame (accurate), Live = smooth video (boxes may lag)
            if synced_view:
                show_frame = display_frame if display_frame is not None else frame
            else:
                show_frame = frame
            display = tracker.draw(show_frame, detections)

            # Show mode indicator
            mode_text = "SYNCED" if synced_view else "LIVE"
            cv2.putText(display, mode_text, (display.shape[1] - 80, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Show CV SKU input buffer (for typing in window)
            if cv_sku_buffer:
                cv2.putText(display, f"SKU: {cv_sku_buffer}_", (20, display.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Resize
            dh, dw = display.shape[:2]
            scale = min(1280/dw, 720/dh, 1.0)
            if scale < 1:
                display = cv2.resize(display, (int(dw*scale), int(dh*scale)))

            cv2.imshow("Scan-Place Tracker", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('i'):
                tracker.print_inventory()
            elif key == ord('r'):
                tracker.set_roi(frame)
            elif key in (ord('+'), ord('=')):  # Increase threshold
                tracker.confidence = min(0.5, tracker.confidence + 0.05)
                print(f"  Threshold: {tracker.confidence:.2f}")
            elif key in (ord('-'), ord('_')):  # Decrease threshold
                tracker.confidence = max(0.05, tracker.confidence - 0.05)
                print(f"  Threshold: {tracker.confidence:.2f}")
            elif key == ord('s'):  # Toggle synced/live view
                synced_view = not synced_view
                print(f"  View: {'SYNCED (accurate)' if synced_view else 'LIVE (smooth)'}")
            elif key in (13, 10):  # Enter (CR or LF)
                if cv_sku_buffer:
                    tracker.queue_sku(cv_sku_buffer)
                    cv_sku_buffer = ""
            elif key == 8:  # Backspace
                cv_sku_buffer = cv_sku_buffer[:-1]
            elif key == 27:  # Escape
                cv_sku_buffer = ""
            elif 32 <= key <= 126:  # Printable characters
                cv_sku_buffer += chr(key).upper()

    finally:
        running = False
        cap.release()
        cv2.destroyAllWindows()
        tracker.print_inventory()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Scan-Then-Place Inventory System")
        print("Usage: python count_scan_place.py .129")
    else:
        run(sys.argv[1])
