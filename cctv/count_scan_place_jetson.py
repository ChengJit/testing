#!/usr/bin/env python3
"""
SCAN-THEN-PLACE Inventory System - JETSON ORIN NANO VERSION
Optimized for Linux/Jetson with CUDA acceleration

Setup on Jetson:
1. Install dependencies:
   pip install opencv-python numpy pillow
   pip install torch torchvision  # Use Jetson-specific wheel from NVIDIA

2. Install GroundingDINO:
   git clone https://github.com/IDEA-Research/GroundingDINO.git
   cd GroundingDINO
   pip install -e .
   mkdir weights && cd weights
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

3. Run:
   python count_scan_place_jetson.py 192.168.122.128
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

# Suppress OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ============ CONFIGURATION ============
CONFIG = {
    # Camera settings
    "rtsp_user": "fasspay",
    "rtsp_pass": "fasspay2025",
    "use_hd": True,              # True=HD stream, False=SD stream

    # Detection settings
    "confidence": 0.20,          # Detection threshold
    "prompt": "cardboard box . brown box . package",
    "min_area": 500,
    "max_area": 500000,
    "detect_interval": 1.0,      # Seconds between detections
    "max_detect_dim": 800,       # Max image size for detection (lower=faster)

    # Tracking settings
    "iou_threshold": 0.15,
    "max_move_distance": 150,
    "sku_timeout": 60,           # Seconds before pending SKU expires

    # Display settings
    "display_width": 1280,
    "display_height": 720,
    "headless": False,           # Set True to run without display
}
# =======================================


def get_jetson_info():
    """Get Jetson device info."""
    info = {"device": "Unknown", "cuda": False}

    # Check for Jetson
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
            if "Orin" in model:
                info["device"] = "Jetson Orin Nano"
            elif "Nano" in model:
                info["device"] = "Jetson Nano"
            elif "Xavier" in model:
                info["device"] = "Jetson Xavier"
            else:
                info["device"] = model
    except:
        info["device"] = "Linux PC"

    # Check CUDA
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

        # Try GStreamer first (better on Jetson), fallback to FFmpeg
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
        """Create GStreamer pipeline for RTSP."""
        # Extract components from RTSP URL
        # rtsp://user:pass@ip:port/stream
        return (
            f"rtspsrc location={url} latency=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )

    def _grab_loop(self):
        """Continuously grab frames, keeping only the latest."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        """Get the latest frame."""
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.cap.release()


class ScanPlaceTracker:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = "cpu"
        self.roi = None

        self.confidence = config["confidence"]
        self.prompt = config["prompt"]
        self.min_area = config["min_area"]
        self.max_area = config["max_area"]

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

    def _download_progress(self, count, block_size, total_size):
        """Show download progress."""
        percent = int(count * block_size * 100 / total_size)
        print(f"\r  Progress: {percent}%", end='', flush=True)

    def load(self):
        """Load detection model."""
        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  PyTorch device: {self.device}")

            if self.device == "cuda":
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

            # Find GroundingDINO
            gdino_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GroundingDINO"),
                os.path.expanduser("~/test/GroundingDINO"),
                os.path.expanduser("~/GroundingDINO"),
                "/opt/GroundingDINO",
                "./GroundingDINO",
            ]

            gdino_path = None
            for path in gdino_paths:
                if os.path.exists(path):
                    gdino_path = path
                    break

            if not gdino_path:
                print("ERROR: GroundingDINO not found!")
                print("Install with: git clone https://github.com/IDEA-Research/GroundingDINO.git")
                return False

            print(f"  GroundingDINO path: {gdino_path}")
            sys.path.insert(0, gdino_path)

            from groundingdino.util.inference import load_model

            config_path = os.path.join(gdino_path, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
            weights_path = os.path.join(gdino_path, "weights", "groundingdino_swint_ogc.pth")

            if not os.path.exists(weights_path):
                print(f"Weights not found, downloading...")
                weights_dir = os.path.dirname(weights_path)
                os.makedirs(weights_dir, exist_ok=True)

                import urllib.request
                url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

                try:
                    print(f"  Downloading from {url}")
                    print(f"  This may take a few minutes...")
                    urllib.request.urlretrieve(url, weights_path, self._download_progress)
                    print("\n  Download complete!")
                except Exception as e:
                    print(f"\nDownload failed: {e}")
                    print("Manual download:")
                    print(f"  wget -P {weights_dir} {url}")
                    return False

            print("  Loading GroundingDINO model...")
            self.model = load_model(config_path, weights_path, device=self.device)
            print("  GroundingDINO ready!")

            # Warmup (optional - skip if causing issues)
            # if self.device == "cuda":
            #     print("  Warming up GPU...")
            #     dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            #     self.detect_boxes(dummy)
            #     print("  GPU warmed up!")

            return True

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
        """Detect boxes."""
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

        # Downscale for speed
        orig_work_h, orig_work_w = work.shape[:2]
        max_dim = self.config["max_detect_dim"]
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

        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=self.prompt,
                box_threshold=self.confidence,
                text_threshold=self.confidence,
                device=self.device
            )

        detections = []
        detect_h, detect_w = work.shape[:2]

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

        detections = self._nms(detections, 0.5)

        # Stabilize count
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
            if remaining < 10:
                cv2.putText(display, "HURRY!", (250, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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
        cv2.putText(display, "S:Sync/Live | +/-:Thresh | R:ROI | I:List | Q:Quit",
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
    print("  Jetson Orin Nano Edition")
    print("=" * 50)

    # Check device
    info = get_jetson_info()
    print(f"\nDevice: {info['device']}")
    print(f"CUDA: {'Available' if info['cuda'] else 'Not available'}")
    if 'gpu' in info:
        print(f"GPU: {info['gpu']}")

    # Build RTSP URL
    stream = "stream1" if CONFIG["use_hd"] else "stream2"
    url = f"rtsp://{CONFIG['rtsp_user']}:{CONFIG['rtsp_pass']}@{camera_ip}:554/{stream}"

    print(f"\nConnecting to {'HD' if CONFIG['use_hd'] else 'SD'} stream...")
    print(f"URL: rtsp://***:***@{camera_ip}:554/{stream}")

    # Initialize tracker
    tracker = ScanPlaceTracker(CONFIG)
    if not tracker.load():
        return

    # Connect to camera
    cap = LatestFrameGrabber(url, use_gstreamer=True)
    time.sleep(1)

    if not cap.isOpened():
        print("Connection failed!")
        return

    print("Connected!")

    # Check headless mode FIRST
    headless = CONFIG.get("headless", False)
    if not headless:
        try:
            cv2.namedWindow("Scan-Place Tracker", cv2.WINDOW_NORMAL)
        except:
            print("No display available, switching to headless mode")
            headless = True

    ret, frame = cap.read()
    if ret:
        print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
        if not headless:
            tracker.set_roi(frame)
        else:
            print("Headless mode: Using full frame (no ROI)")

    if headless:
        print("\n" + "=" * 40)
        print("  HEADLESS MODE - No display")
        print("=" * 40)
        print("Commands:")
        print("  <SKU>        - Queue SKU for placement")
        print("  list         - Show inventory")
        print("  status       - Show current status")
        print("  thresh 0.25  - Set detection threshold")
        print("  quit         - Exit")
        print("=" * 40 + "\n")

    # Detection thread
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
                print(f"  Detection: {elapsed:.2f}s, {len(dets)} boxes")
            except Exception as e:
                print(f"Detection error: {e}")
        detecting = False

    last_detect = 0
    detect_interval = CONFIG["detect_interval"]

    # SKU input via stdin
    sku_queue = queue.Queue()
    running = True

    def stdin_reader():
        while running:
            try:
                line = input()
                cmd = line.strip().lower()

                # Handle commands
                if cmd == 'quit' or cmd == 'q':
                    sku_queue.put('__QUIT__')
                    break
                elif cmd == 'list' or cmd == 'i':
                    sku_queue.put('__LIST__')
                elif cmd == 'status':
                    sku_queue.put('__STATUS__')
                elif cmd.startswith('thresh '):
                    sku_queue.put(f'__THRESH__{cmd[7:]}')
                elif line.strip():
                    sku = line.strip().upper()
                    sku_queue.put(sku)
            except EOFError:
                break
            except:
                pass

    input_thread = threading.Thread(target=stdin_reader, daemon=True)
    input_thread.start()
    print("\n>>> Ready! Scan barcode or type SKU here <<<\n")

    synced_view = True
    cv_sku_buffer = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Check barcode input and commands
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
                            tracker.confidence = max(0.05, min(0.5, val))
                            print(f"  Threshold: {tracker.confidence:.2f}")
                        except:
                            print("  Usage: thresh 0.25")
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

            # Display (skip if headless)
            if not headless:
                if synced_view:
                    show_frame = display_frame if display_frame is not None else frame
                else:
                    show_frame = frame
                display = tracker.draw(show_frame, detections)

                mode_text = "SYNCED" if synced_view else "LIVE"
                cv2.putText(display, mode_text, (display.shape[1] - 80, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if cv_sku_buffer:
                    cv2.putText(display, f"SKU: {cv_sku_buffer}_", (20, display.shape[0] - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # Resize for display
                dh, dw = display.shape[:2]
                scale = min(CONFIG["display_width"]/dw, CONFIG["display_height"]/dh, 1.0)
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
                elif key in (ord('+'), ord('=')):
                    tracker.confidence = min(0.5, tracker.confidence + 0.05)
                    print(f"  Threshold: {tracker.confidence:.2f}")
                elif key in (ord('-'), ord('_')):
                    tracker.confidence = max(0.05, tracker.confidence - 0.05)
                    print(f"  Threshold: {tracker.confidence:.2f}")
                elif key == ord('s'):
                    synced_view = not synced_view
                    print(f"  View: {'SYNCED' if synced_view else 'LIVE'}")
                elif key in (13, 10):
                    if cv_sku_buffer:
                        tracker.queue_sku(cv_sku_buffer)
                        cv_sku_buffer = ""
                elif key == 8:
                    cv_sku_buffer = cv_sku_buffer[:-1]
                elif key == 27:
                    cv_sku_buffer = ""
                elif 32 <= key <= 126:
                    cv_sku_buffer += chr(key).upper()
            else:
                # Headless mode - just sleep a bit
                time.sleep(0.01)

    finally:
        running = False
        cap.release()
        cv2.destroyAllWindows()
        tracker.print_inventory()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_scan_place_jetson.py <camera_ip>")
        print("Example: python count_scan_place_jetson.py 192.168.122.128")
        print("\nConfiguration can be edited at the top of this file.")
    else:
        run(sys.argv[1])
