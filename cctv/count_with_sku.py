#!/usr/bin/env python3
"""
Box Counter + SKU Reader
Counts boxes AND reads barcodes/SKUs automatically!
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


class BoxSKUCounter:
    def __init__(self):
        self.model = None
        self.barcode_reader = None
        self.roi = None
        self.confidence = 0.25
        self.stable_count = 0
        self.count_history = []

        # SKU tracking
        self.box_skus = {}  # box_id -> SKU
        self.sku_log = []   # History of detected SKUs

    def load(self):
        """Load detection model and barcode reader."""
        try:
            # Try YOLO-World first (faster)
            from ultralytics import YOLO
            print("Loading YOLO-World...")
            self.model = YOLO("yolov8s-world.pt")
            self.model.set_classes(["cardboard box", "box", "package", "carton"])
            print("  YOLO ready!")
        except:
            print("YOLO-World not available, trying GroundingDINO...")
            return False

        # Load barcode reader
        try:
            from pyzbar import pyzbar
            self.barcode_reader = pyzbar
            print("  Barcode reader ready!")
        except ImportError:
            print("  Barcode reader not installed!")
            print("  Install: pip install pyzbar")
            print("  Also need: https://github.com/NaturalHistoryMuseum/ZBarWin64/releases")
            self.barcode_reader = None

        return True

    def detect_boxes(self, frame):
        """Detect boxes in frame."""
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

        results = self.model(work, conf=self.confidence, verbose=False)

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    'x1': bx1 + offset[0], 'y1': by1 + offset[1],
                    'x2': bx2 + offset[0], 'y2': by2 + offset[1],
                    'score': float(box.conf[0]),
                    'sku': None,
                    'barcode_img': None
                })

        # Stabilize count
        self.count_history.append(len(detections))
        if len(self.count_history) > 5:
            self.count_history.pop(0)
        self.stable_count = sorted(self.count_history)[len(self.count_history)//2]

        return detections

    def read_barcodes(self, frame, detections):
        """Read barcodes from detected boxes."""
        if self.barcode_reader is None:
            return detections

        for det in detections:
            # Crop box region (with padding)
            pad = 10
            x1 = max(0, det['x1'] - pad)
            y1 = max(0, det['y1'] - pad)
            x2 = min(frame.shape[1], det['x2'] + pad)
            y2 = min(frame.shape[0], det['y2'] + pad)

            box_img = frame[y1:y2, x1:x2]

            # Try to read barcode
            try:
                # Preprocess for better barcode detection
                gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)

                # Try multiple thresholds
                for thresh_method in [None, cv2.THRESH_BINARY, cv2.THRESH_OTSU]:
                    if thresh_method:
                        _, processed = cv2.threshold(gray, 127, 255, thresh_method)
                    else:
                        processed = gray

                    barcodes = self.barcode_reader.decode(processed)

                    if barcodes:
                        barcode = barcodes[0]
                        sku = barcode.data.decode('utf-8')
                        det['sku'] = sku
                        det['barcode_type'] = barcode.type

                        # Log new SKU
                        if sku not in [s['sku'] for s in self.sku_log]:
                            self.sku_log.append({
                                'sku': sku,
                                'time': datetime.now().isoformat(),
                                'type': barcode.type
                            })
                            print(f"  NEW SKU: {sku} ({barcode.type})")
                        break

            except Exception as e:
                pass

        return detections

    def draw(self, frame, detections):
        """Draw boxes and SKUs."""
        display = frame.copy()
        h, w = display.shape[:2]

        # ROI
        if self.roi:
            rx1, ry1 = int(self.roi[0]*w), int(self.roi[1]*h)
            rx2, ry2 = int(self.roi[2]*w), int(self.roi[3]*h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

        # Boxes
        for i, det in enumerate(detections):
            color = (0, 255, 0) if det['sku'] else (0, 200, 255)

            cv2.rectangle(display, (det['x1'], det['y1']),
                         (det['x2'], det['y2']), color, 2)

            # Label
            label = f"{i+1}"
            if det['sku']:
                label = f"{det['sku'][:15]}"  # Truncate long SKUs

            cv2.putText(display, label, (det['x1']+5, det['y1']+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Info panel
        cv2.rectangle(display, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.putText(display, f"Boxes: {self.stable_count}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        skus_found = sum(1 for d in detections if d['sku'])
        cv2.putText(display, f"SKUs read: {skus_found}/{len(detections)}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.putText(display, f"Total unique SKUs: {len(self.sku_log)}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Help
        cv2.rectangle(display, (0, h-25), (w, h), (30, 30, 30), -1)
        cv2.putText(display, "R:ROI  L:List SKUs  +/-:Conf  Q:Quit",
                   (10, h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

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

    def print_sku_list(self):
        """Print all detected SKUs."""
        print("\n" + "=" * 40)
        print("  DETECTED SKUs")
        print("=" * 40)
        if not self.sku_log:
            print("  No SKUs detected yet")
        for s in self.sku_log:
            print(f"  {s['sku']} ({s['type']}) - {s['time']}")
        print("=" * 40 + "\n")


def run(camera_ip):
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2"

    print("=" * 50)
    print("  BOX COUNTER + SKU READER")
    print("=" * 50)

    counter = BoxSKUCounter()
    if not counter.load():
        return

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Connection failed!")
        return

    print("Connected!")

    ret, frame = cap.read()
    if ret:
        counter.set_roi(frame)

    cv2.namedWindow("Box+SKU Counter", cv2.WINDOW_NORMAL)

    last_detect = 0
    detections = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect periodically
            if time.time() - last_detect >= 1.0:
                detections = counter.detect_boxes(frame)
                detections = counter.read_barcodes(frame, detections)
                last_detect = time.time()

            display = counter.draw(frame, detections)

            dh, dw = display.shape[:2]
            scale = min(1280/dw, 720/dh, 1.0)
            if scale < 1:
                display = cv2.resize(display, (int(dw*scale), int(dh*scale)))

            cv2.imshow("Box+SKU Counter", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.set_roi(frame)
            elif key == ord('l'):
                counter.print_sku_list()
            elif key in [ord('+'), ord('=')]:
                counter.confidence = min(0.9, counter.confidence + 0.05)
                print(f"Conf: {counter.confidence:.2f}")
            elif key == ord('-'):
                counter.confidence = max(0.1, counter.confidence - 0.05)
                print(f"Conf: {counter.confidence:.2f}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        counter.print_sku_list()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Box Counter + SKU Reader")
        print("Usage: python count_with_sku.py .129")
    else:
        run(sys.argv[1])
