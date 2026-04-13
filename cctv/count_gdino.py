#!/usr/bin/env python3
"""
GroundingDINO Box Counter - Fast & Accurate!
Uses text prompts to find boxes. Better than OWL!
"""

import cv2
import numpy as np
import time
import os
import sys
import threading

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Add GroundingDINO to path
GDINO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GroundingDINO")
sys.path.insert(0, GDINO_PATH)


class GDINOCounter:
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.roi = None
        self.baseline = 0
        self.last_count = 0

        # Detection settings - SMARTER PROMPTS
        self.prompt = "cardboard box . brown box . package box . shipping box"
        self.box_threshold = 0.25  # Tuned for your boxes!
        self.text_threshold = 0.25
        self.min_area = 800       # Ignore tiny detections
        self.max_area = 80000

        # Stabilization - keep history to smooth flickering
        self.count_history = []
        self.history_size = 5  # Remember last 5 counts
        self.stable_count = 0

        # Calibration - user teaches correct count
        self.calibrated = False
        self.calibrated_count = 0
        self.calibrated_boxes = []  # Remember box positions
        self.change_threshold = 0.3  # How much change to accept new count

        # SMART TRACKING - remember boxes across frames
        self.tracked_boxes = []  # Persistent tracked boxes
        self.box_hits = {}       # How many times each box seen (id -> count)
        self.next_box_id = 0
        self.min_hits = 4        # Box must appear 4 times to count (more strict)
        self.max_age = 8         # Keep box longer if temporarily missed
        self.iou_threshold = 0.25 # Match threshold (lower = easier match)
        self.smooth_factor = 0.7  # Position smoothing (0=no smooth, 1=full smooth)

        # AUTO-COLLECT: Save when count changes
        self.auto_collect = True
        self.last_saved_count = -1
        self.collect_dir = os.path.join(os.path.dirname(__file__), "auto_collected")
        self.review_dir = os.path.join(os.path.dirname(__file__), "to_review")
        os.makedirs(self.collect_dir, exist_ok=True)
        os.makedirs(self.review_dir, exist_ok=True)
        self.min_save_interval = 5  # Seconds between auto-saves
        self.last_save_time = 0
        self.total_collected = len([f for f in os.listdir(self.collect_dir) if f.endswith('.jpg')])

        # HIGH-CONFIDENCE AUTO-APPROVE: Skip review for confident detections
        self.auto_approve = True
        self.auto_approve_threshold = 0.6  # High confidence = auto-approve to training
        self.training_dir = os.path.join(os.path.dirname(__file__), "training_data")

    def load(self):
        """Load GroundingDINO model."""
        try:
            import torch

            print("Loading GroundingDINO...")

            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"  GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                print("  Using CPU")

            # Paths
            config_path = os.path.join(GDINO_PATH, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
            weights_path = os.path.join(GDINO_PATH, "weights", "groundingdino_swint_ogc.pth")

            # Download weights if missing
            if not os.path.exists(weights_path):
                print("  Downloading weights (first time only)...")
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                import urllib.request
                url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
                urllib.request.urlretrieve(url, weights_path)
                print("  Downloaded!")

            # Load model
            from groundingdino.util.inference import load_model
            self.model = load_model(config_path, weights_path, device=self.device)

            print("  GroundingDINO ready!")
            return True

        except Exception as e:
            print(f"Error: {e}")
            print("\nInstall: pip install supervision torch torchvision")
            return False

    def detect(self, frame):
        """Detect boxes."""
        if self.model is None:
            return []

        import torch
        from PIL import Image
        import groundingdino.datasets.transforms as T
        from groundingdino.util.inference import predict

        h, w = frame.shape[:2]

        # Apply ROI
        if self.roi:
            x1 = int(self.roi[0] * w)
            y1 = int(self.roi[1] * h)
            x2 = int(self.roi[2] * w)
            y2 = int(self.roi[3] * h)
            work = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            work = frame
            offset = (0, 0)

        # Use full resolution of stream for max accuracy
        work_h, work_w = work.shape[:2]
        scale = 1.0

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
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device
        )

        # Convert to detections
        detections = []
        img_h, img_w = work.shape[:2]

        for box, score, label in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box.tolist()
            bx1 = int((cx - bw/2) * img_w / scale) + offset[0]
            by1 = int((cy - bh/2) * img_h / scale) + offset[1]
            bx2 = int((cx + bw/2) * img_w / scale) + offset[0]
            by2 = int((cy + bh/2) * img_h / scale) + offset[1]

            area = (bx2 - bx1) * (by2 - by1)
            if self.min_area <= area <= self.max_area:
                detections.append({
                    'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2,
                    'score': float(score),
                    'label': label
                })

        # Remove overlapping detections (NMS)
        detections = self._nms(detections, 0.5)

        # SMART TRACKING - match detections to tracked boxes
        tracked_detections = self._update_tracking(detections)

        # Count only confirmed boxes (seen enough times)
        confirmed_count = len([d for d in tracked_detections if d.get('confirmed', False)])

        # Also keep raw count for display
        raw_count = len(detections)
        self.count_history.append(confirmed_count)
        if len(self.count_history) > self.history_size:
            self.count_history.pop(0)

        # Use median for extra stability
        sorted_counts = sorted(self.count_history)
        median_count = sorted_counts[len(sorted_counts) // 2]

        # If calibrated, lock to calibrated count unless big change
        if self.calibrated:
            diff = abs(median_count - self.calibrated_count)
            if diff >= 2 and all(abs(c - self.calibrated_count) >= 2 for c in self.count_history[-3:]):
                self.stable_count = median_count
                print(f"  Count changed: {self.calibrated_count} -> {median_count}")
            else:
                self.stable_count = self.calibrated_count
        else:
            self.stable_count = median_count

        self.last_count = self.stable_count
        self._last_frame = frame.copy()  # For auto-save
        self._last_detections = tracked_detections
        return tracked_detections

    def try_auto_save(self, frame, detections):
        """Called from main loop to auto-save on count change."""
        confirmed = [d for d in detections if d.get('confirmed', False)]
        self.auto_save_on_change(frame, confirmed, len(confirmed))

    def _nms(self, detections, threshold=0.5):
        """Remove overlapping boxes, keep best score."""
        if len(detections) <= 1:
            return detections

        # Sort by score
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)

        keep = []
        for det in detections:
            is_overlap = False
            for kept in keep:
                if self._iou(det, kept) > threshold:
                    is_overlap = True
                    break
            if not is_overlap:
                keep.append(det)

        return keep

    def _iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _update_tracking(self, detections):
        """Match detections to tracked boxes, update hits."""
        # Age all tracked boxes
        for tb in self.tracked_boxes:
            tb['age'] += 1

        # Match detections to existing tracked boxes
        matched_tracks = set()
        matched_dets = set()

        for i, det in enumerate(detections):
            best_iou = 0
            best_track = None

            for j, tb in enumerate(self.tracked_boxes):
                if j in matched_tracks:
                    continue
                iou = self._iou(det, tb)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track = j

            if best_track is not None:
                # Update tracked box with SMOOTHING (reduce jitter)
                tb = self.tracked_boxes[best_track]
                s = self.smooth_factor
                tb['x1'] = int(tb['x1'] * s + det['x1'] * (1-s))
                tb['y1'] = int(tb['y1'] * s + det['y1'] * (1-s))
                tb['x2'] = int(tb['x2'] * s + det['x2'] * (1-s))
                tb['y2'] = int(tb['y2'] * s + det['y2'] * (1-s))
                tb['hits'] += 1
                tb['age'] = 0
                tb['score'] = max(tb['score'], det['score'])  # Keep best score
                matched_tracks.add(best_track)
                matched_dets.add(i)

        # Create new tracked boxes for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self.tracked_boxes.append({
                    'id': self.next_box_id,
                    'x1': det['x1'], 'y1': det['y1'],
                    'x2': det['x2'], 'y2': det['y2'],
                    'hits': 1,
                    'age': 0,
                    'score': det['score'],
                    'label': det.get('label', 'box')
                })
                self.next_box_id += 1

        # Remove old tracked boxes
        self.tracked_boxes = [tb for tb in self.tracked_boxes if tb['age'] <= self.max_age]

        # Return tracked boxes with confirmed status
        result = []
        for tb in self.tracked_boxes:
            box = {
                'x1': tb['x1'], 'y1': tb['y1'],
                'x2': tb['x2'], 'y2': tb['y2'],
                'score': tb['score'],
                'label': tb.get('label', 'box'),
                'hits': tb['hits'],
                'confirmed': tb['hits'] >= self.min_hits  # Only count if seen enough!
            }
            result.append(box)

        return result

    def calibrate(self, detections, frame=None):
        """User says current count is correct! Also save for training."""
        self.calibrated = True
        self.calibrated_count = len(detections)
        self.calibrated_boxes = [(d['x1'], d['y1'], d['x2'], d['y2']) for d in detections]
        self.stable_count = self.calibrated_count

        # Auto-save training data!
        if frame is not None and len(detections) > 0:
            self._save_training_data(frame, detections)

        print(f"CALIBRATED! Locked to {self.calibrated_count} boxes")
        return self.calibrated_count

    def _save_training_data(self, frame, detections):
        """Save frame + labels for training YOLO later."""
        import os

        # Create training folder
        train_dir = os.path.join(os.path.dirname(__file__), "training_data")
        img_dir = os.path.join(train_dir, "images")
        lbl_dir = os.path.join(train_dir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # Count existing files
        existing = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        fname = f"box_{existing:04d}"

        # Save image
        img_path = os.path.join(img_dir, f"{fname}.jpg")
        cv2.imwrite(img_path, frame)

        # Save YOLO format labels (class x_center y_center width height)
        h, w = frame.shape[:2]
        lbl_path = os.path.join(lbl_dir, f"{fname}.txt")
        with open(lbl_path, 'w') as f:
            for d in detections:
                # Convert to YOLO format (normalized)
                x_center = ((d['x1'] + d['x2']) / 2) / w
                y_center = ((d['y1'] + d['y2']) / 2) / h
                box_w = (d['x2'] - d['x1']) / w
                box_h = (d['y2'] - d['y1']) / h
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

        print(f"  Saved training data: {fname} ({existing + 1} samples)")

    def auto_save_on_change(self, frame, detections, current_count):
        """Auto-save when count changes. High-confidence = auto-approve!"""
        if not self.auto_collect:
            return

        now = time.time()

        # Only save if count changed AND enough time passed
        if (current_count != self.last_saved_count and
            current_count > 0 and
            now - self.last_save_time >= self.min_save_interval):

            # Check if all detections are high-confidence
            high_conf = all(d.get('score', 0) >= self.auto_approve_threshold for d in detections)

            if high_conf and self.auto_approve and len(detections) >= 3:
                # HIGH CONFIDENCE - Save directly to training (no review needed!)
                self._save_training_data(frame, detections)
                self.last_saved_count = current_count
                self.last_save_time = now
                print(f"  [AUTO-APPROVED] High confidence - saved to training!")
            else:
                # Lower confidence - save for review
                timestamp = int(now)
                img_name = f"auto_{timestamp}_count{current_count}.jpg"
                lbl_name = f"auto_{timestamp}_count{current_count}.txt"

                img_path = os.path.join(self.collect_dir, img_name)
                lbl_path = os.path.join(self.collect_dir, lbl_name)

                cv2.imwrite(img_path, frame)

                h, w = frame.shape[:2]
                with open(lbl_path, 'w') as f:
                    for d in detections:
                        if d.get('confirmed', True):
                            x_center = ((d['x1'] + d['x2']) / 2) / w
                            y_center = ((d['y1'] + d['y2']) / 2) / h
                            box_w = (d['x2'] - d['x1']) / w
                            box_h = (d['y2'] - d['y1']) / h
                            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

                self.last_saved_count = current_count
                self.last_save_time = now
                self.total_collected += 1
                print(f"  [AUTO] Needs review: {img_name}")

    def uncalibrate(self):
        """Reset calibration."""
        self.calibrated = False
        self.calibrated_count = 0
        self.calibrated_boxes = []
        print("Calibration reset!")

    def set_roi(self, frame):
        """Draw ROI - resized to fit screen."""
        h, w = frame.shape[:2]

        # Resize to fit screen (max 1280x720 for drawing)
        max_w, max_h = 1280, 720
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            display_frame = frame.copy()
            scale = 1.0

        pts = []
        drawing = False
        temp = display_frame.copy()

        def mouse(event, x, y, flags, param):
            nonlocal pts, drawing, temp
            if event == cv2.EVENT_LBUTTONDOWN:
                pts = [(x, y)]
                drawing = True
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp = display_frame.copy()
                cv2.rectangle(temp, pts[0], (x, y), (0, 255, 255), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                pts.append((x, y))
                drawing = False

        win = "Draw ROI - ENTER confirm, ESC skip"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, mouse)

        while True:
            cv2.imshow(win, temp)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and len(pts) == 2:
                break
            elif key == 27:
                cv2.destroyWindow(win)
                return

        cv2.destroyWindow(win)

        # Convert back to original scale
        disp_h, disp_w = display_frame.shape[:2]
        self.roi = (
            min(pts[0][0], pts[1][0]) / disp_w,
            min(pts[0][1], pts[1][1]) / disp_h,
            max(pts[0][0], pts[1][0]) / disp_w,
            max(pts[0][1], pts[1][1]) / disp_h,
        )
        print("ROI set!")

    def draw(self, frame, detections):
        """Draw results."""
        display = frame.copy()
        h, w = display.shape[:2]

        # ROI
        if self.roi:
            rx1 = int(self.roi[0] * w)
            ry1 = int(self.roi[1] * h)
            rx2 = int(self.roi[2] * w)
            ry2 = int(self.roi[3] * h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

        # Detections - green=confirmed, yellow=pending
        confirmed_num = 0
        for d in detections:
            if d.get('confirmed', False):
                confirmed_num += 1
                color = (0, 255, 0)  # Green = confirmed
                cv2.rectangle(display, (d['x1'], d['y1']), (d['x2'], d['y2']), color, 2)
                cv2.putText(display, f"{confirmed_num}", (d['x1'] + 5, d['y1'] + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                color = (0, 200, 255)  # Yellow = still checking
                cv2.rectangle(display, (d['x1'], d['y1']), (d['x2'], d['y2']), color, 1)

        # Count - show stable count (big) and raw count (small)
        cv2.rectangle(display, (10, 10), (220, 115), (0, 0, 0), -1)

        # Show calibration status
        if self.calibrated:
            cv2.putText(display, "LOCKED", (150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        confirmed = len([d for d in detections if d.get('confirmed', False)])
        pending = len(detections) - confirmed

        cv2.putText(display, f"Boxes: {self.stable_count}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(display, f"(confirmed:{confirmed} pending:{pending})", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        if self.baseline > 0:
            change = self.stable_count - self.baseline
            color = (0, 255, 0) if change >= 0 else (0, 0, 255)
            cv2.putText(display, f"Change: {change:+d}", (20, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Info
        cv2.putText(display, f"Thresh:{self.box_threshold:.2f} Hits:{self.min_hits}", (w - 180, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Auto-collect status
        ac_color = (0, 255, 0) if self.auto_collect else (100, 100, 100)
        ac_text = f"Auto:{self.total_collected}" if self.auto_collect else "Auto:OFF"
        cv2.putText(display, ac_text, (w - 180, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ac_color, 1)

        # Help
        cv2.rectangle(display, (0, h - 25), (w, h), (40, 40, 40), -1)
        cv2.putText(display, "C:Lock T:Save A:AutoCollect U:Unlock B:Base R:ROI +/-:Thresh Q:Quit",
                   (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)

        return display


def run(camera_ip):
    """Run counter."""
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2?rtsp_transport=tcp"  # SD stream, stable

    print("=" * 40)
    print("  GROUNDINGDINO BOX COUNTER")
    print("=" * 40)

    counter = GDINOCounter()
    if not counter.load():
        return

    print(f"\nConnecting to {camera_ip}...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("No connect!")
        return

    print("Connected!")

    cap.grab()
    ret, frame = cap.retrieve()
    if ret:
        counter.set_roi(frame)

    print("\nControls:")
    print("  C = LOCK count + SAVE training data!")
    print("  T = Save MORE training samples")
    print("  U = Unlock (reset calibration)")
    print("  +/- = Adjust detection threshold")
    print("  [/] = Adjust min hits (how many times box must appear)")
    print("  B = Set baseline  R = ROI  Q = Quit")
    print("\nSMART TRACKING: Green=confirmed, Yellow=still checking")
    print("-" * 40)

    win = "GroundingDINO Counter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Threading for smooth video
    detections = []
    detecting = False
    detect_frame = None
    detect_result = None
    last_detect = 0
    detect_interval = 3.0  # Slower but more stable

    def detect_thread():
        nonlocal detect_result, detecting
        if detect_frame is not None:
            print("Detecting...", end=" ", flush=True)
            start = time.time()
            dets = counter.detect(detect_frame)
            print(f"{len(dets)} boxes ({time.time()-start:.1f}s)")
            detect_result = dets
        detecting = False

    error_count = 0
    max_errors = 30

    try:
        while True:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                error_count += 1
                if error_count > max_errors:
                    print("Too many errors, reconnecting...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    error_count = 0
                continue
            error_count = 0  # Reset on good frame

            # Background detection
            if not detecting and time.time() - last_detect >= detect_interval:
                detect_frame = frame.copy()
                detecting = True
                last_detect = time.time()
                threading.Thread(target=detect_thread, daemon=True).start()

            if detect_result is not None:
                detections = detect_result
                detect_result = None
                # Auto-save when count changes
                counter.try_auto_save(frame, detections)

            display = counter.draw(frame, detections)

            # Resize display to fit screen
            dh, dw = display.shape[:2]
            max_dw, max_dh = 1280, 720
            disp_scale = min(max_dw / dw, max_dh / dh, 1.0)
            if disp_scale < 1.0:
                display = cv2.resize(display, (int(dw * disp_scale), int(dh * disp_scale)))

            cv2.imshow(win, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Lock current count as correct + save training data!
                counter.calibrate(detections, frame)
            elif key == ord('u'):
                # Unlock/reset calibration
                counter.uncalibrate()
            elif key == ord('b'):
                counter.baseline = counter.last_count
                print(f"Baseline: {counter.baseline}")
            elif key == ord('r'):
                counter.set_roi(frame)
            elif key == ord('+') or key == ord('='):
                counter.box_threshold = min(0.9, counter.box_threshold + 0.05)
                print(f"Threshold: {counter.box_threshold:.2f}")
            elif key == ord('-'):
                counter.box_threshold = max(0.1, counter.box_threshold - 0.05)
                print(f"Threshold: {counter.box_threshold:.2f}")
            elif key == ord('['):
                counter.min_hits = max(1, counter.min_hits - 1)
                print(f"Min hits: {counter.min_hits} (box must be seen {counter.min_hits}x to count)")
            elif key == ord(']'):
                counter.min_hits = min(10, counter.min_hits + 1)
                print(f"Min hits: {counter.min_hits} (box must be seen {counter.min_hits}x to count)")
            elif key == ord('t'):
                # Save training sample (without recalibrating)
                if len(detections) > 0:
                    counter._save_training_data(frame, detections)
                else:
                    print("No detections to save!")
            elif key == ord('s'):
                fname = f"gdino_{int(time.time())}.jpg"
                cv2.imwrite(fname, display)
                print(f"Saved: {fname}")
            elif key == ord('a'):
                counter.auto_collect = not counter.auto_collect
                status = "ON" if counter.auto_collect else "OFF"
                print(f"Auto-collect: {status}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("GroundingDINO Box Counter")
        print("=" * 30)
        print()
        print("Fast & accurate box detection!")
        print()
        print("Usage: python count_gdino.py <camera_ip>")
        print("Example: python count_gdino.py .129")
    else:
        run(sys.argv[1])
