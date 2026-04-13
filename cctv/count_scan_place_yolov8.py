#!/usr/bin/env python3
"""
SCAN-THEN-PLACE Inventory System - YOLOv8 VERSION
Optimized for speed on Jetson Orin Nano

Setup on Jetson:
1. Install dependencies:
   pip install ultralytics opencv-python numpy

2. Run with pretrained model (detects generic boxes):
   python count_scan_place_yolov8.py 192.168.122.128

3. Train on your own boxes (see bottom of file for instructions)
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

# Suppress warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ============ CONFIGURATION ============
CONFIG = {
    # Camera settings
    "rtsp_user": "fasspay",
    "rtsp_pass": "fasspay2025",
    "use_hd": True,

    # Detection settings
    "confidence": 0.15,
    "model_path": "yolov8n.pt",  # Use custom model: "runs/detect/train/weights/best.pt"
    "detect_classes": None,      # None=all, or [0] for person, [39,41,42,64] for bottles/cups/boxes
    "detect_interval": 0.3,      # Can be much lower with YOLOv8!
    "input_size": 640,           # YOLO input size

    # Box filtering
    "min_area": 500,
    "max_area": 500000,

    # Tracking settings
    "iou_threshold": 0.15,
    "max_move_distance": 150,
    "sku_timeout": 60,

    # Display settings
    "display_width": 1280,
    "display_height": 720,
    "headless": True,

    # Logging
    "log_file": "detection_log.csv",
}
# =======================================


def get_jetson_info():
    """Get Jetson device info."""
    info = {"device": "Unknown", "cuda": False}
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
            if "Orin" in model:
                info["device"] = "Jetson Orin Nano"
            elif "Nano" in model:
                info["device"] = "Jetson Nano"
            else:
                info["device"] = model
    except:
        info["device"] = "Linux PC"

    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except:
        pass
    return info


class LatestFrameGrabber:
    """Grabs latest frame from RTSP, discarding old buffered frames."""

    def __init__(self, url, use_gstreamer=True):
        self.url = url
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        if use_gstreamer:
            gst_pipeline = self._make_gstreamer_pipeline(url)
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                print("  GStreamer failed, trying FFmpeg...")
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()

    def _make_gstreamer_pipeline(self, url):
        return (
            f"rtspsrc location={url} latency=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )

    def _grab_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.cap.release()


class YOLOv8Tracker:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = "cpu"
        self.roi = None

        self.confidence = config["confidence"]
        self.min_area = config["min_area"]
        self.max_area = config["max_area"]
        self.detect_classes = config["detect_classes"]

        # Count stabilization
        self.count_history = []
        self.stable_count = 0

        # Box tracking
        self.tracked_boxes = []
        self.next_box_id = 1
        self.iou_threshold = config["iou_threshold"]
        self.max_move_distance = config["max_move_distance"]

        # Assignment feedback
        self.last_assignment = None

        # Scan-then-place queue
        self.pending_skus = deque()
        self.sku_timeout = config["sku_timeout"]

        # Inventory
        self.inventory = {}
        self.total_scanned = 0
        self.total_placed = 0

    def load(self):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  PyTorch device: {self.device}")

            if self.device == "cuda":
                print(f"  GPU: {torch.cuda.get_device_name(0)}")

            model_path = self.config["model_path"]
            print(f"  Loading YOLOv8 model: {model_path}")

            self.model = YOLO(model_path)

            # Warmup
            print("  Warming up model...")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)
            print("  YOLOv8 ready!")

            return True

        except ImportError:
            print("ERROR: ultralytics not installed!")
            print("Install with: pip install ultralytics")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def queue_sku(self, sku):
        """Add SKU to pending queue."""
        self.pending_skus.append({
            'sku': sku,
            'time': time.time()
        })
        self.total_scanned += 1
        print(f"\n  [SCANNED] {sku} - place box now!")

    def get_pending_info(self):
        """Get pending SKU with time remaining."""
        now = time.time()
        while self.pending_skus:
            if now - self.pending_skus[0]['time'] > self.sku_timeout:
                expired = self.pending_skus.popleft()
                print(f"  [EXPIRED] {expired['sku']}")
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
        """Detect boxes using YOLOv8."""
        if self.model is None:
            return []

        h, w = frame.shape[:2]
        work = frame
        offset = (0, 0)

        if self.roi:
            x1, y1 = int(self.roi[0]*w), int(self.roi[1]*h)
            x2, y2 = int(self.roi[2]*w), int(self.roi[3]*h)
            work = frame[y1:y2, x1:x2]
            offset = (x1, y1)

        # Run YOLOv8 inference
        results = self.model.predict(
            work,
            conf=self.confidence,
            imgsz=self.config["input_size"],
            classes=self.detect_classes,
            verbose=False
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls = int(box.cls[0])

                # Adjust for ROI offset
                bx1 = int(x1) + offset[0]
                by1 = int(y1) + offset[1]
                bx2 = int(x2) + offset[0]
                by2 = int(y2) + offset[1]

                area = (bx2 - bx1) * (by2 - by1)
                if self.min_area <= area <= self.max_area:
                    detections.append({
                        'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2,
                        'score': score,
                        'class': cls
                    })

        # Stabilize count
        self.count_history.append(len(detections))
        if len(self.count_history) > 5:
            self.count_history.pop(0)
        self.stable_count = sorted(self.count_history)[len(self.count_history)//2]

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

    def _centroid_dist(self, box1, box2):
        cx1 = (box1['x1'] + box1['x2']) / 2
        cy1 = (box1['y1'] + box1['y2']) / 2
        cx2 = (box2['x1'] + box2['x2']) / 2
        cy2 = (box2['y1'] + box2['y2']) / 2
        return ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5

    def track_and_assign(self, detections):
        """Track boxes and assign pending SKUs to NEW boxes."""
        for tb in self.tracked_boxes:
            tb['missing'] += 1

        matched_tracks = set()
        matched_dets = set()

        for i, det in enumerate(detections):
            best_score = 0
            best_idx = None

            for j, tb in enumerate(self.tracked_boxes):
                if j in matched_tracks:
                    continue

                iou = self._iou(det, tb)
                if iou >= self.iou_threshold:
                    score = iou + 1.0
                    if score > best_score:
                        best_score = score
                        best_idx = j
                else:
                    dist = self._centroid_dist(det, tb)
                    if dist < self.max_move_distance:
                        score = 1.0 - (dist / self.max_move_distance)
                        if score > best_score:
                            best_score = score
                            best_idx = j

            if best_idx is not None:
                tb = self.tracked_boxes[best_idx]
                tb['x1'], tb['y1'] = det['x1'], det['y1']
                tb['x2'], tb['y2'] = det['x2'], det['y2']
                tb['missing'] = 0

                det['box_id'] = tb['id']
                det['sku'] = tb['sku']
                det['is_new'] = False

                matched_tracks.add(best_idx)
                matched_dets.add(i)

        for i, det in enumerate(detections):
            if i not in matched_dets:
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
                    print(f"\n  *** [ASSIGNED] Box #{self.next_box_id} = {pending_sku} ***")
                    self.last_assignment = (pending_sku, time.time())

                self.next_box_id += 1

        self.tracked_boxes = [tb for tb in self.tracked_boxes if tb['missing'] <= 5]

        return detections

    def _get_zone(self, det):
        cx = (det['x1'] + det['x2']) / 2
        cy = (det['y1'] + det['y2']) / 2
        col = chr(ord('A') + int(cx / 200) % 4)
        row = str(int(cy / 200) % 4 + 1)
        return f"{col}{row}"

    def draw(self, frame, detections):
        """Draw boxes and inventory."""
        display = frame.copy()
        h, w = display.shape[:2]

        if self.roi:
            rx1, ry1 = int(self.roi[0]*w), int(self.roi[1]*h)
            rx2, ry2 = int(self.roi[2]*w), int(self.roi[3]*h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

        assigned = 0
        unassigned = 0

        for det in detections:
            if det.get('sku'):
                color = (0, 255, 0)
                assigned += 1
                label = det['sku'][:16]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(display, (det['x1'], det['y1']-th-8),
                             (det['x1']+tw+8, det['y1']), (0, 180, 0), -1)
                cv2.putText(display, label, (det['x1']+4, det['y1']-4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            else:
                color = (0, 100, 255)
                unassigned += 1
                cv2.putText(display, f"#{det.get('box_id', '?')}", (det['x1']+5, det['y1']+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            thickness = 4 if det.get('is_new') else 2
            cv2.rectangle(display, (det['x1'], det['y1']),
                         (det['x2'], det['y2']), color, thickness)

        # Info panel
        cv2.rectangle(display, (10, 10), (420, 140), (0, 0, 0), -1)
        cv2.putText(display, f"Boxes: {self.stable_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display, f"Tagged: {assigned}  Untagged: {unassigned}  Thresh: {self.confidence:.2f}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        pending, remaining = self.get_pending_info()
        if pending:
            cv2.putText(display, f"NEXT: {pending}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"Place box now ({int(remaining)}s)", (20, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        else:
            cv2.putText(display, "Ready - scan SKU", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Assignment confirmation
        if self.last_assignment:
            sku, assign_time = self.last_assignment
            if time.time() - assign_time < 3.0:
                cv2.rectangle(display, (w//2 - 200, h//2 - 50), (w//2 + 200, h//2 + 50), (0, 180, 0), -1)
                cv2.putText(display, f"ASSIGNED: {sku}", (w//2 - 150, h//2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                self.last_assignment = None

        # Help
        cv2.rectangle(display, (0, h-30), (w, h), (30, 30, 30), -1)
        cv2.putText(display, "YOLOv8 | +/-:Thresh | R:ROI | I:List | Q:Quit",
                   (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return display

    def set_roi(self, frame):
        """Draw ROI interactively."""
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


def run(camera_ip):
    """Main run function."""
    print("=" * 50)
    print("  SCAN-THEN-PLACE INVENTORY SYSTEM")
    print("  YOLOv8 Edition (Fast)")
    print("=" * 50)

    info = get_jetson_info()
    print(f"\nDevice: {info['device']}")
    print(f"CUDA: {'Available' if info['cuda'] else 'Not available'}")

    # Build RTSP URL
    stream = "stream1" if CONFIG["use_hd"] else "stream2"
    url = f"rtsp://{CONFIG['rtsp_user']}:{CONFIG['rtsp_pass']}@{camera_ip}:554/{stream}"

    print(f"\nConnecting to {'HD' if CONFIG['use_hd'] else 'SD'} stream...")

    # Initialize tracker
    tracker = YOLOv8Tracker(CONFIG)
    if not tracker.load():
        return

    # Connect to camera
    cap = LatestFrameGrabber(url, use_gstreamer=True)
    time.sleep(1)

    if not cap.isOpened():
        print("Connection failed!")
        return

    print("Connected!")

    headless = CONFIG.get("headless", False)
    if not headless:
        try:
            cv2.namedWindow("YOLOv8 Tracker", cv2.WINDOW_NORMAL)
        except:
            print("No display available, switching to headless mode")
            headless = True

    ret, frame = cap.read()
    if ret:
        print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
        if not headless:
            tracker.set_roi(frame)

    if headless:
        print("\n" + "=" * 40)
        print("  HEADLESS MODE - YOLOv8")
        print("=" * 40)

    # Initialize log file
    log_file = CONFIG.get("log_file")
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"\n# YOLOv8 Session: {datetime.now().isoformat()}\n")
            f.write("timestamp,detection_time_s,box_count,tagged_count,untagged_count\n")

    # Detection state
    detections = []
    detecting = False
    detect_frame = None
    detect_result = None
    display_frame = None

    def detect_thread():
        nonlocal detect_result, detecting, display_frame
        if detect_frame is not None:
            try:
                start = time.time()
                dets = tracker.detect_boxes(detect_frame)
                dets = tracker.track_and_assign(dets)
                detect_result = dets
                display_frame = detect_frame.copy()
                elapsed = time.time() - start

                tagged = sum(1 for d in dets if d.get('sku'))
                untagged = len(dets) - tagged

                print(f"  Detection: {elapsed:.3f}s, {len(dets)} boxes (tagged:{tagged}, untagged:{untagged})")

                if log_file:
                    with open(log_file, "a") as f:
                        f.write(f"{datetime.now().isoformat()},{elapsed:.3f},{len(dets)},{tagged},{untagged}\n")
            except Exception as e:
                print(f"Detection error: {e}")
        detecting = False

    last_detect = 0
    detect_interval = CONFIG["detect_interval"]

    # SKU input
    sku_queue = queue.Queue()
    running = True

    def stdin_reader():
        while running:
            try:
                line = input()
                cmd = line.strip().lower()
                if cmd in ('quit', 'q'):
                    sku_queue.put('__QUIT__')
                    break
                elif cmd in ('list', 'i'):
                    sku_queue.put('__LIST__')
                elif cmd == 'status':
                    sku_queue.put('__STATUS__')
                elif cmd.startswith('thresh '):
                    sku_queue.put(f'__THRESH__{cmd[7:]}')
                elif line.strip():
                    sku_queue.put(line.strip().upper())
            except EOFError:
                break
            except:
                pass

    input_thread = threading.Thread(target=stdin_reader, daemon=True)
    input_thread.start()
    print("\n>>> Ready! Scan barcode or type SKU <<<\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Process commands
            should_quit = False
            try:
                while True:
                    item = sku_queue.get_nowait()
                    if item == '__QUIT__':
                        should_quit = True
                    elif item == '__LIST__':
                        tracker.print_inventory()
                    elif item == '__STATUS__':
                        print(f"\n  Boxes: {tracker.stable_count} | Tagged: {len(tracker.inventory)} | Pending: {len(tracker.pending_skus)}\n")
                    elif item.startswith('__THRESH__'):
                        try:
                            val = float(item[10:])
                            tracker.confidence = max(0.05, min(0.9, val))
                            print(f"  Threshold: {tracker.confidence:.2f}")
                        except:
                            pass
                    else:
                        tracker.queue_sku(item)
            except queue.Empty:
                pass

            if should_quit:
                break

            # Background detection
            if not detecting and time.time() - last_detect >= detect_interval:
                detect_frame = frame.copy()
                detecting = True
                last_detect = time.time()
                threading.Thread(target=detect_thread, daemon=True).start()

            if detect_result is not None:
                detections = detect_result
                detect_result = None

            # Display
            if not headless:
                show_frame = display_frame if display_frame is not None else frame
                display = tracker.draw(show_frame, detections)

                dh, dw = display.shape[:2]
                scale = min(CONFIG["display_width"]/dw, CONFIG["display_height"]/dh, 1.0)
                if scale < 1:
                    display = cv2.resize(display, (int(dw*scale), int(dh*scale)))

                cv2.imshow("YOLOv8 Tracker", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('i'):
                    tracker.print_inventory()
                elif key == ord('r'):
                    tracker.set_roi(frame)
                elif key in (ord('+'), ord('=')):
                    tracker.confidence = min(0.9, tracker.confidence + 0.05)
                    print(f"  Threshold: {tracker.confidence:.2f}")
                elif key in (ord('-'), ord('_')):
                    tracker.confidence = max(0.05, tracker.confidence - 0.05)
                    print(f"  Threshold: {tracker.confidence:.2f}")
            else:
                time.sleep(0.01)

    finally:
        running = False
        cap.release()
        cv2.destroyAllWindows()
        tracker.print_inventory()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_scan_place_yolov8.py <camera_ip>")
        print("\n" + "=" * 50)
        print("  HOW TO TRAIN ON YOUR OWN BOXES")
        print("=" * 50)
        print("""
1. Collect images:
   - Save ~50-100 images of your boxes
   - Put them in: dataset/images/

2. Label images (use LabelImg or Roboflow):
   - Create bounding boxes around each box
   - Save labels in YOLO format to: dataset/labels/

3. Create dataset.yaml:
   path: dataset
   train: images
   val: images
   names:
     0: box

4. Train:
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   model.train(data='dataset.yaml', epochs=50, imgsz=640)

5. Use trained model:
   Edit CONFIG["model_path"] = "runs/detect/train/weights/best.pt"
""")
    else:
        run(sys.argv[1])
