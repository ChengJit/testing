#!/usr/bin/env python3
"""
SAM (Segment Anything) + YOLO for box counting
SAM gives pixel-perfect masks!
"""

import cv2
import numpy as np
import os
import time

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


class SAMCounter:
    def __init__(self):
        self.sam = None
        self.yolo = None
        self.roi = None
        self.confidence = 0.3
        self.show_masks = True
        self.stable_count = 0
        self.calibrated = False
        self.calibrated_count = 0
        self.count_history = []

    def load(self):
        """Load SAM and YOLO models."""
        try:
            from ultralytics import YOLO, SAM

            print("Loading YOLO-World for detection...")
            self.yolo = YOLO("yolov8s-world.pt")
            self.yolo.set_classes(["cardboard box", "box", "package"])
            print("  YOLO ready!")

            print("Loading SAM (downloads ~100MB first time)...")
            self.sam = SAM("sam_b.pt")
            print("  SAM ready!")
            return True

        except Exception as e:
            print(f"Error: {e}")
            print("\nInstall: pip install ultralytics")
            return False

    def detect_and_segment(self, frame):
        """Detect with YOLO, segment with SAM."""
        if self.yolo is None:
            return []

        h, w = frame.shape[:2]
        work = frame
        offset = (0, 0)

        if self.roi:
            x1, y1 = int(self.roi[0]*w), int(self.roi[1]*h)
            x2, y2 = int(self.roi[2]*w), int(self.roi[3]*h)
            work = frame[y1:y2, x1:x2]
            offset = (x1, y1)

        # YOLO detection
        results = self.yolo(work, conf=self.confidence, verbose=False)

        detections = []
        boxes = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    'x1': bx1 + offset[0], 'y1': by1 + offset[1],
                    'x2': bx2 + offset[0], 'y2': by2 + offset[1],
                    'score': float(box.conf[0]), 'mask': None
                })
                boxes.append([bx1, by1, bx2, by2])

        # SAM segmentation
        if boxes and self.show_masks and self.sam:
            try:
                sam_results = self.sam(work, bboxes=boxes, verbose=False)
                for i, r in enumerate(sam_results):
                    if r.masks is not None and i < len(detections):
                        mask = r.masks.data[0].cpu().numpy()
                        detections[i]['mask'] = (mask * 255).astype(np.uint8)
            except:
                pass

        # Stabilize
        self.count_history.append(len(detections))
        if len(self.count_history) > 5:
            self.count_history.pop(0)
        median = sorted(self.count_history)[len(self.count_history)//2]
        self.stable_count = self.calibrated_count if self.calibrated else median

        return detections

    def draw(self, frame, detections):
        """Draw boxes and masks."""
        display = frame.copy()
        h, w = display.shape[:2]

        colors = [(255,100,100), (100,255,100), (100,100,255),
                  (255,255,100), (255,100,255), (100,255,255)]

        for i, d in enumerate(detections):
            color = colors[i % len(colors)]

            # Draw mask
            if d.get('mask') is not None and self.show_masks:
                mask = d['mask']
                if self.roi:
                    rx1, ry1 = int(self.roi[0]*w), int(self.roi[1]*h)
                    rx2, ry2 = int(self.roi[2]*w), int(self.roi[3]*h)
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    mh, mw = mask.shape[:2]
                    full_mask[ry1:ry1+mh, rx1:rx1+mw] = mask
                    mask = full_mask

                if mask.shape[:2] == (h, w):
                    overlay = display.copy()
                    overlay[mask > 127] = color
                    display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

            cv2.rectangle(display, (d['x1'], d['y1']), (d['x2'], d['y2']), color, 2)
            cv2.putText(display, str(i+1), (d['x1']+5, d['y1']+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ROI
        if self.roi:
            rx1, ry1 = int(self.roi[0]*w), int(self.roi[1]*h)
            rx2, ry2 = int(self.roi[2]*w), int(self.roi[3]*h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0,255,255), 2)

        # Info
        cv2.rectangle(display, (10,10), (300,90), (0,0,0), -1)
        cv2.putText(display, f"Boxes: {self.stable_count}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 2)
        status = f"SAM:{'ON' if self.show_masks else 'OFF'} | raw:{len(detections)}"
        cv2.putText(display, status, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

        cv2.rectangle(display, (0, h-25), (w, h), (30,30,30), -1)
        cv2.putText(display, "M:Masks C:Lock +/-:Conf Q:Quit",
                   (10, h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        return display

    def set_roi(self, frame):
        h, w = frame.shape[:2]
        scale = min(1280/w, 720/h, 1.0)
        display = cv2.resize(frame, (int(w*scale), int(h*scale))) if scale < 1 else frame.copy()

        pts, drawing, temp = [], False, display.copy()

        def mouse(event, x, y, flags, param):
            nonlocal pts, drawing, temp
            if event == cv2.EVENT_LBUTTONDOWN:
                pts, drawing = [(x,y)], True
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp = display.copy()
                cv2.rectangle(temp, pts[0], (x,y), (0,255,255), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                pts.append((x,y))
                drawing = False

        cv2.namedWindow("ROI")
        cv2.setMouseCallback("ROI", mouse)
        while True:
            cv2.imshow("ROI", temp)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and len(pts) == 2: break
            elif key == 27: cv2.destroyWindow("ROI"); return
        cv2.destroyWindow("ROI")

        dh, dw = display.shape[:2]
        self.roi = (min(pts[0][0],pts[1][0])/dw, min(pts[0][1],pts[1][1])/dh,
                   max(pts[0][0],pts[1][0])/dw, max(pts[0][1],pts[1][1])/dh)


def run(camera_ip):
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2"

    print("=" * 50)
    print("  SAM + YOLO BOX COUNTER")
    print("  Pixel-perfect segmentation!")
    print("=" * 50)

    counter = SAMCounter()
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

    cv2.namedWindow("SAM Counter", cv2.WINDOW_NORMAL)

    last_detect, detections = 0, []

    try:
        while True:
            ret, frame = cap.read()
            if not ret: continue

            if time.time() - last_detect >= 1.5:
                start = time.time()
                detections = counter.detect_and_segment(frame)
                print(f"Detected {len(detections)} in {time.time()-start:.2f}s")
                last_detect = time.time()

            display = counter.draw(frame, detections)
            dh, dw = display.shape[:2]
            scale = min(1280/dw, 720/dh, 1.0)
            if scale < 1: display = cv2.resize(display, (int(dw*scale), int(dh*scale)))

            cv2.imshow("SAM Counter", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'): counter.show_masks = not counter.show_masks
            elif key == ord('c'): counter.calibrated = True; counter.calibrated_count = len(detections)
            elif key == ord('u'): counter.calibrated = False
            elif key in [ord('+'), ord('=')]: counter.confidence = min(0.9, counter.confidence+0.05)
            elif key == ord('-'): counter.confidence = max(0.1, counter.confidence-0.05)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("SAM + YOLO Counter - Pixel perfect masks!")
        print("Usage: python count_sam.py .129")
    else:
        run(sys.argv[1])
