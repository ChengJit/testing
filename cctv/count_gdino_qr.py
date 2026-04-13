#!/usr/bin/env python3
"""
GroundingDINO Box Counter + QR/Barcode SKU Reader
Uses HD stream (3K) for better barcode reading!
"""

import cv2
import numpy as np
import os
import sys
import time
import threading
from datetime import datetime

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Add GroundingDINO to path
GDINO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GroundingDINO")
sys.path.insert(0, GDINO_PATH)


class GDinoQRCounter:
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.roi = None
        self.confidence = 0.25
        self.stable_count = 0
        self.count_history = []

        # Detection settings
        self.prompt = "cardboard box . brown box . package"
        self.min_area = 1000
        self.max_area = 500000  # Bigger for HD

        # QR/Barcode reader
        self.qr_reader = None
        self.barcode_reader = None

        # SKU tracking with PERSISTENT MEMORY
        self.sku_log = []
        self.last_frame = None

        # BOX TRACKING - remember boxes across frames
        self.tracked_boxes = []  # List of tracked boxes with IDs and SKUs
        self.next_box_id = 1
        self.iou_threshold = 0.3  # Match threshold
        self.max_missing_frames = 10  # Keep box memory for N frames if not detected

        # SMART SELF-TRAINING
        self.auto_train = True
        self.training_dir = os.path.join(os.path.dirname(__file__), "training_data")
        self.auto_collect_dir = os.path.join(os.path.dirname(__file__), "auto_collected")
        os.makedirs(os.path.join(self.training_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.training_dir, "labels"), exist_ok=True)
        os.makedirs(self.auto_collect_dir, exist_ok=True)

        self.last_saved_count = -1
        self.last_save_time = 0
        self.min_save_interval = 5  # seconds between saves
        self.auto_approve_threshold = 0.5  # High confidence = auto-approve
        self.samples_collected = self._count_samples()
        self.samples_approved = self._count_approved()

    def _count_samples(self):
        """Count collected samples."""
        try:
            return len([f for f in os.listdir(self.auto_collect_dir) if f.endswith('.jpg')])
        except:
            return 0

    def _count_approved(self):
        """Count approved training samples."""
        try:
            img_dir = os.path.join(self.training_dir, "images")
            return len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        except:
            return 0

    def auto_save_training(self, frame, detections):
        """Smart self-training: auto-save when count changes."""
        if not self.auto_train or len(detections) == 0:
            return

        now = time.time()
        current_count = len(detections)

        # Only save if count changed and enough time passed
        if (current_count != self.last_saved_count and
            now - self.last_save_time >= self.min_save_interval):

            # Check if all detections are high confidence
            avg_conf = sum(d['score'] for d in detections) / len(detections)
            high_conf = avg_conf >= self.auto_approve_threshold

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            h, w = frame.shape[:2]

            if high_conf:
                # HIGH CONFIDENCE - Auto-approve to training data!
                img_path = os.path.join(self.training_dir, "images", f"auto_{timestamp}.jpg")
                lbl_path = os.path.join(self.training_dir, "labels", f"auto_{timestamp}.txt")

                cv2.imwrite(img_path, frame)
                self._save_yolo_labels(lbl_path, detections, w, h)

                self.samples_approved += 1
                print(f"  [AUTO-APPROVED] conf:{avg_conf:.2f} → training data ({self.samples_approved} total)")
            else:
                # Lower confidence - save for review
                img_path = os.path.join(self.auto_collect_dir, f"review_{timestamp}_c{current_count}.jpg")
                lbl_path = os.path.join(self.auto_collect_dir, f"review_{timestamp}_c{current_count}.txt")

                cv2.imwrite(img_path, frame)
                self._save_yolo_labels(lbl_path, detections, w, h)

                self.samples_collected += 1
                print(f"  [NEEDS REVIEW] conf:{avg_conf:.2f} → auto_collected ({self.samples_collected} total)")

            self.last_saved_count = current_count
            self.last_save_time = now

    def _save_yolo_labels(self, path, detections, img_w, img_h):
        """Save detections in YOLO format."""
        with open(path, 'w') as f:
            for d in detections:
                x_center = ((d['x1'] + d['x2']) / 2) / img_w
                y_center = ((d['y1'] + d['y2']) / 2) / img_h
                box_w = (d['x2'] - d['x1']) / img_w
                box_h = (d['y2'] - d['y1']) / img_h
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

    def manual_save(self, frame, detections):
        """Manual save (press C) - always goes to training data."""
        if len(detections) == 0:
            print("No detections to save!")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        h, w = frame.shape[:2]

        img_path = os.path.join(self.training_dir, "images", f"manual_{timestamp}.jpg")
        lbl_path = os.path.join(self.training_dir, "labels", f"manual_{timestamp}.txt")

        cv2.imwrite(img_path, frame)
        self._save_yolo_labels(lbl_path, detections, w, h)

        self.samples_approved += 1
        print(f"  [SAVED] {len(detections)} boxes → training data ({self.samples_approved} total)")

    def load(self):
        """Load models."""
        try:
            import torch
            from groundingdino.util.inference import load_model

            print("Loading GroundingDINO...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Device: {self.device}")

            config_path = os.path.join(GDINO_PATH, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
            weights_path = os.path.join(GDINO_PATH, "weights", "groundingdino_swint_ogc.pth")

            self.model = load_model(config_path, weights_path, device=self.device)
            print("  GroundingDINO ready!")

        except Exception as e:
            print(f"GroundingDINO error: {e}")
            return False

        # Load QR reader (OpenCV built-in)
        try:
            self.qr_reader = cv2.QRCodeDetector()
            print("  QR reader ready!")
        except:
            print("  QR reader not available")

        # Load barcode reader (pyzbar)
        try:
            from pyzbar import pyzbar
            self.barcode_reader = pyzbar
            print("  Barcode reader ready!")
        except:
            print("  Barcode reader not installed (pip install pyzbar)")

        return True

    def detect_boxes(self, frame):
        """Detect boxes with GroundingDINO."""
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

        # Transform
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        image_pil = Image.fromarray(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))
        image_tensor, _ = transform(image_pil, None)

        # Predict
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=self.prompt,
            box_threshold=self.confidence,
            text_threshold=self.confidence,
            device=self.device
        )

        # Convert to detections
        detections = []
        work_h, work_w = work.shape[:2]

        for box, score, label in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box.tolist()
            bx1 = int((cx - bw/2) * work_w) + offset[0]
            by1 = int((cy - bh/2) * work_h) + offset[1]
            bx2 = int((cx + bw/2) * work_w) + offset[0]
            by2 = int((cy + bh/2) * work_h) + offset[1]

            area = (bx2 - bx1) * (by2 - by1)
            if self.min_area <= area <= self.max_area:
                detections.append({
                    'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2,
                    'score': float(score),
                    'sku': None,
                    'qr_found': False
                })

        # NMS
        detections = self._nms(detections, 0.5)

        # Stabilize
        self.count_history.append(len(detections))
        if len(self.count_history) > 5:
            self.count_history.pop(0)
        self.stable_count = sorted(self.count_history)[len(self.count_history)//2]

        self.last_frame = frame.copy()
        return detections

    def _nms(self, detections, threshold):
        """Non-maximum suppression."""
        if len(detections) <= 1:
            return detections

        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []

        for det in detections:
            overlap = False
            for kept in keep:
                # Calculate IoU
                x1 = max(det['x1'], kept['x1'])
                y1 = max(det['y1'], kept['y1'])
                x2 = min(det['x2'], kept['x2'])
                y2 = min(det['y2'], kept['y2'])

                if x2 > x1 and y2 > y1:
                    inter = (x2-x1) * (y2-y1)
                    area1 = (det['x2']-det['x1']) * (det['y2']-det['y1'])
                    area2 = (kept['x2']-kept['x1']) * (kept['y2']-kept['y1'])
                    iou = inter / (area1 + area2 - inter)
                    if iou > threshold:
                        overlap = True
                        break

            if not overlap:
                keep.append(det)

        return keep

    def read_qr_barcodes(self, frame, detections):
        """Read QR codes and barcodes - SKIP already scanned boxes!"""
        scanned_count = 0
        new_scans = 0

        for det in detections:
            # SKIP if already scanned!
            if det.get('already_scanned') or det.get('sku'):
                scanned_count += 1
                continue

            # Crop box region with padding
            pad = 20
            x1 = max(0, det['x1'] - pad)
            y1 = max(0, det['y1'] - pad)
            x2 = min(frame.shape[1], det['x2'] + pad)
            y2 = min(frame.shape[0], det['y2'] + pad)

            box_img = frame[y1:y2, x1:x2]
            if box_img.size == 0:
                continue

            sku_found = None

            # Try QR code first (OpenCV)
            if self.qr_reader:
                try:
                    data, bbox, _ = self.qr_reader.detectAndDecode(box_img)
                    if data:
                        sku_found = data
                        det['qr_found'] = True
                        self._log_sku(data, "QR")
                except:
                    pass

            # Try barcode (pyzbar) - try normal AND inverted
            if not sku_found and self.barcode_reader:
                try:
                    gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)

                    # Try multiple preprocessing methods
                    variants = [
                        gray,                              # Original
                        cv2.equalizeHist(gray),           # Enhanced contrast
                        255 - gray,                        # Inverted (white on black)
                        255 - cv2.equalizeHist(gray),     # Inverted + enhanced
                    ]

                    for img in variants:
                        barcodes = self.barcode_reader.decode(img)
                        if barcodes:
                            bc = barcodes[0]
                            sku_found = bc.data.decode('utf-8')
                            det['qr_found'] = (bc.type == 'QRCODE')
                            self._log_sku(sku_found, bc.type)
                            break
                except:
                    pass

            # Update detection AND tracked box
            if sku_found:
                det['sku'] = sku_found
                det['already_scanned'] = True
                new_scans += 1

                # Update tracked box memory
                if 'box_id' in det:
                    self.update_box_sku(det['box_id'], sku_found)

        if new_scans > 0:
            print(f"  [SCAN] {new_scans} new SKUs! (total scanned: {scanned_count + new_scans})")

        return detections

    def _log_sku(self, sku, code_type):
        """Log new SKU."""
        if sku not in [s['sku'] for s in self.sku_log]:
            self.sku_log.append({
                'sku': sku,
                'type': code_type,
                'time': datetime.now().strftime("%H:%M:%S")
            })
            print(f"  [NEW SKU] {sku} ({code_type})")

    def _iou(self, box1, box2):
        """Calculate IoU between two boxes."""
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

    def track_boxes(self, detections):
        """Match detections to tracked boxes, keep SKUs persistent."""
        # Age all tracked boxes
        for tb in self.tracked_boxes:
            tb['missing'] += 1

        matched_tracks = set()
        matched_dets = set()

        # Match detections to existing tracked boxes
        for i, det in enumerate(detections):
            best_iou = 0
            best_track_idx = None

            for j, tb in enumerate(self.tracked_boxes):
                if j in matched_tracks:
                    continue
                iou = self._iou(det, tb)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_idx = j

            if best_track_idx is not None:
                # Update tracked box position
                tb = self.tracked_boxes[best_track_idx]
                tb['x1'] = det['x1']
                tb['y1'] = det['y1']
                tb['x2'] = det['x2']
                tb['y2'] = det['y2']
                tb['score'] = det['score']
                tb['missing'] = 0

                # Keep existing SKU!
                det['sku'] = tb['sku']
                det['box_id'] = tb['id']
                det['already_scanned'] = tb['sku'] is not None

                matched_tracks.add(best_track_idx)
                matched_dets.add(i)

        # Create new tracked boxes for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                new_box = {
                    'id': self.next_box_id,
                    'x1': det['x1'], 'y1': det['y1'],
                    'x2': det['x2'], 'y2': det['y2'],
                    'score': det['score'],
                    'sku': None,
                    'missing': 0
                }
                self.tracked_boxes.append(new_box)
                det['box_id'] = self.next_box_id
                det['sku'] = None
                det['already_scanned'] = False
                self.next_box_id += 1

        # Remove old tracked boxes (missing too long)
        self.tracked_boxes = [tb for tb in self.tracked_boxes if tb['missing'] <= self.max_missing_frames]

        return detections

    def update_box_sku(self, box_id, sku):
        """Update SKU for a tracked box."""
        for tb in self.tracked_boxes:
            if tb['id'] == box_id:
                tb['sku'] = sku
                break

    def draw(self, frame, detections):
        """Draw results."""
        display = frame.copy()
        h, w = display.shape[:2]

        # ROI
        if self.roi:
            rx1, ry1 = int(self.roi[0]*w), int(self.roi[1]*h)
            rx2, ry2 = int(self.roi[2]*w), int(self.roi[3]*h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 255), 3)

        # Boxes
        scanned = 0
        unscanned = 0

        for det in detections:
            box_id = det.get('box_id', '?')

            if det.get('sku'):
                color = (0, 255, 0)  # Green = has SKU
                scanned += 1

                # SKU label with background
                label = det['sku'][:18]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(display, (det['x1'], det['y1']-th-8),
                             (det['x1']+tw+8, det['y1']), (0, 200, 0), -1)
                cv2.putText(display, label, (det['x1']+4, det['y1']-4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            else:
                color = (0, 150, 255)  # Orange = no SKU yet
                unscanned += 1

                # Show box ID for unscanned
                cv2.putText(display, f"#{box_id}", (det['x1']+5, det['y1']+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display, "scanning...", (det['x1']+5, det['y1']+50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            cv2.rectangle(display, (det['x1'], det['y1']),
                         (det['x2'], det['y2']), color, 2)

        # Info panel
        cv2.rectangle(display, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.putText(display, f"Boxes: {self.stable_count}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Scanned vs unscanned
        cv2.putText(display, f"Scanned: {scanned}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Pending: {unscanned}", (180, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)

        cv2.putText(display, f"Unique SKUs: {len(self.sku_log)} | Tracked: {len(self.tracked_boxes)}",
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.putText(display, f"Conf:{self.confidence:.2f}",
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Training status
        train_color = (0, 255, 0) if self.auto_train else (100, 100, 100)
        train_text = f"Train:ON ({self.samples_approved})" if self.auto_train else "Train:OFF"
        cv2.putText(display, train_text, (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, train_color, 1)
        cv2.putText(display, f"Review:{self.samples_collected}", (w - 200, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Help
        cv2.rectangle(display, (0, h-30), (w, h), (30, 30, 30), -1)
        cv2.putText(display, "C:Save T:Train A:AutoTrain L:SKUs R:ROI +/-:Conf Q:Quit",
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

        cv2.namedWindow("Draw ROI - ENTER to confirm")
        cv2.setMouseCallback("Draw ROI - ENTER to confirm", mouse)

        while True:
            cv2.imshow("Draw ROI - ENTER to confirm", temp)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and len(pts) == 2:
                break
            elif key == 27:
                cv2.destroyWindow("Draw ROI - ENTER to confirm")
                return

        cv2.destroyWindow("Draw ROI - ENTER to confirm")
        dh, dw = display.shape[:2]
        self.roi = (min(pts[0][0], pts[1][0])/dw, min(pts[0][1], pts[1][1])/dh,
                   max(pts[0][0], pts[1][0])/dw, max(pts[0][1], pts[1][1])/dh)

    def print_sku_list(self):
        """Print all detected SKUs."""
        print("\n" + "=" * 50)
        print("  ALL DETECTED SKUs")
        print("=" * 50)
        if not self.sku_log:
            print("  No SKUs detected yet")
            print("  Make sure QR codes are visible and not blurry!")
        else:
            for i, s in enumerate(self.sku_log, 1):
                print(f"  {i}. {s['sku']} ({s['type']}) @ {s['time']}")
        print("=" * 50 + "\n")


def run(camera_ip):
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"

    # Use stream1 (HD) for better QR reading!
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream1"

    print("=" * 50)
    print("  GROUNDINGDINO + QR/BARCODE READER")
    print("  Using HD stream for better scanning!")
    print("=" * 50)

    counter = GDinoQRCounter()
    if not counter.load():
        return

    print(f"\nConnecting to HD stream: {camera_ip}...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("HD stream failed, trying SD...")
        url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Connection failed!")
        return

    print("Connected!")

    ret, frame = cap.read()
    if ret:
        print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
        counter.set_roi(frame)

    cv2.namedWindow("GDINO + QR", cv2.WINDOW_NORMAL)

    # Threading for detection
    detections = []
    detecting = False
    detect_frame = None
    detect_result = None

    def detect_thread():
        nonlocal detect_result, detecting
        if detect_frame is not None:
            dets = counter.detect_boxes(detect_frame)
            dets = counter.track_boxes(dets)  # Track boxes, keep SKUs!
            dets = counter.read_qr_barcodes(detect_frame, dets)  # Only scan unscanned
            detect_result = dets
        detecting = False

    last_detect = 0
    detect_interval = 2.0  # Every 2 seconds

    try:
        while True:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                continue

            # Background detection
            if not detecting and time.time() - last_detect >= detect_interval:
                detect_frame = frame.copy()
                detecting = True
                last_detect = time.time()
                threading.Thread(target=detect_thread, daemon=True).start()

            if detect_result is not None:
                detections = detect_result
                detect_result = None
                # Auto-save for training
                counter.auto_save_training(frame, detections)

            display = counter.draw(frame, detections)

            # Resize for display
            dh, dw = display.shape[:2]
            scale = min(1280/dw, 720/dh, 1.0)
            if scale < 1:
                display = cv2.resize(display, (int(dw*scale), int(dh*scale)))

            cv2.imshow("GDINO + QR", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.set_roi(frame)
            elif key == ord('l'):
                counter.print_sku_list()
            elif key == ord('s'):
                # Force scan now
                print("Scanning...")
                detections = counter.detect_boxes(frame)
                detections = counter.read_qr_barcodes(frame, detections)
            elif key in [ord('+'), ord('=')]:
                counter.confidence = min(0.9, counter.confidence + 0.05)
                print(f"Conf: {counter.confidence:.2f}")
            elif key == ord('-'):
                counter.confidence = max(0.1, counter.confidence - 0.05)
                print(f"Conf: {counter.confidence:.2f}")
            elif key == ord('c'):
                # Manual save to training
                counter.manual_save(frame, detections)
            elif key == ord('a'):
                # Toggle auto-train
                counter.auto_train = not counter.auto_train
                print(f"Auto-train: {'ON' if counter.auto_train else 'OFF'}")
            elif key == ord('t'):
                # Trigger training
                print("\n" + "=" * 40)
                print("  TRAINING YOLO MODEL")
                print("=" * 40)
                print(f"Approved samples: {counter.samples_approved}")
                print(f"To train, run: python train_from_dino.py")
                print("=" * 40 + "\n")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        counter.print_sku_list()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("GroundingDINO + QR/Barcode Reader")
        print("Usage: python count_gdino_qr.py .129")
    else:
        run(sys.argv[1])
