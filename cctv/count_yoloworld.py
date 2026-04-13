#!/usr/bin/env python3
"""
YOLO-World Box Counter - Fast text-based detection!
Like GroundingDINO but 10x faster.
"""

import cv2
import os
import time

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


class YOLOWorldCounter:
    def __init__(self):
        self.model = None
        self.roi = None
        self.baseline = 0
        self.confidence = 0.3
        self.classes = ["cardboard box", "box", "package", "carton"]

        # Stabilization
        self.count_history = []
        self.history_size = 5
        self.stable_count = 0
        self.calibrated = False
        self.calibrated_count = 0

    def load(self):
        """Load YOLO-World model."""
        try:
            from ultralytics import YOLO

            print("Loading YOLO-World...")
            # yolov8s-world = small, fast, text-prompted
            self.model = YOLO("yolov8s-world.pt")

            # Set classes to detect
            self.model.set_classes(self.classes)

            print(f"  Detecting: {self.classes}")
            print("  YOLO-World ready!")
            return True

        except Exception as e:
            print(f"Error: {e}")
            print("\nInstall: pip install ultralytics")
            return False

    def detect(self, frame):
        """Detect boxes."""
        if self.model is None:
            return []

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

        # Run YOLO-World
        results = self.model(work, conf=self.confidence, verbose=False)

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({
                    'x1': bx1 + offset[0],
                    'y1': by1 + offset[1],
                    'x2': bx2 + offset[0],
                    'y2': by2 + offset[1],
                    'score': conf,
                    'label': self.classes[cls] if cls < len(self.classes) else 'box'
                })

        # Stabilize
        raw_count = len(detections)
        self.count_history.append(raw_count)
        if len(self.count_history) > self.history_size:
            self.count_history.pop(0)

        median = sorted(self.count_history)[len(self.count_history) // 2]

        if self.calibrated:
            if abs(median - self.calibrated_count) >= 2:
                if all(abs(c - self.calibrated_count) >= 2 for c in self.count_history[-3:]):
                    self.stable_count = median
            else:
                self.stable_count = self.calibrated_count
        else:
            self.stable_count = median

        return detections

    def calibrate(self, count):
        self.calibrated = True
        self.calibrated_count = count
        self.stable_count = count
        print(f"LOCKED: {count} boxes")

    def set_roi(self, frame):
        """Draw ROI."""
        h, w = frame.shape[:2]
        scale = min(1280/w, 720/h, 1.0)
        if scale < 1.0:
            display = cv2.resize(frame, (int(w*scale), int(h*scale)))
        else:
            display = frame.copy()

        pts = []
        drawing = False
        temp = display.copy()

        def mouse(event, x, y, flags, param):
            nonlocal pts, drawing, temp
            if event == cv2.EVENT_LBUTTONDOWN:
                pts = [(x, y)]
                drawing = True
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp = display.copy()
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
        dh, dw = display.shape[:2]
        self.roi = (
            min(pts[0][0], pts[1][0]) / dw,
            min(pts[0][1], pts[1][1]) / dh,
            max(pts[0][0], pts[1][0]) / dw,
            max(pts[0][1], pts[1][1]) / dh,
        )

    def draw(self, frame, detections):
        """Draw results."""
        display = frame.copy()
        h, w = display.shape[:2]

        if self.roi:
            rx1, ry1 = int(self.roi[0]*w), int(self.roi[1]*h)
            rx2, ry2 = int(self.roi[2]*w), int(self.roi[3]*h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

        for i, d in enumerate(detections):
            cv2.rectangle(display, (d['x1'], d['y1']), (d['x2'], d['y2']), (0, 255, 0), 2)
            cv2.putText(display, f"{i+1}", (d['x1']+5, d['y1']+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Count display
        cv2.rectangle(display, (10, 10), (250, 80), (0, 0, 0), -1)
        cv2.putText(display, f"Boxes: {self.stable_count}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(display, f"(raw: {len(detections)})", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        if self.calibrated:
            cv2.putText(display, "LOCKED", (180, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # FPS/Conf
        cv2.putText(display, f"Conf:{self.confidence:.2f}", (w-100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Help
        cv2.rectangle(display, (0, h-25), (w, h), (40, 40, 40), -1)
        cv2.putText(display, "C:Lock U:Unlock R:ROI +/-:Conf Q:Quit",
                   (10, h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return display


def run(camera_ip):
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2"

    print("=" * 50)
    print("  YOLO-WORLD BOX COUNTER")
    print("  Fast text-based detection!")
    print("=" * 50)

    counter = YOLOWorldCounter()
    if not counter.load():
        return

    print(f"\nConnecting to {camera_ip}...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Connection failed!")
        return

    print("Connected!")

    ret, frame = cap.read()
    if ret:
        counter.set_roi(frame)

    win = "YOLO-World Counter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps_time = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect
            start = time.time()
            detections = counter.detect(frame)
            detect_time = time.time() - start

            # FPS
            fps = 0.9 * fps + 0.1 * (1.0 / (time.time() - fps_time + 0.001))
            fps_time = time.time()

            display = counter.draw(frame, detections)

            # Show detect time
            cv2.putText(display, f"FPS:{fps:.1f} Det:{detect_time*1000:.0f}ms",
                       (display.shape[1]-200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Resize
            dh, dw = display.shape[:2]
            scale = min(1280/dw, 720/dh, 1.0)
            if scale < 1.0:
                display = cv2.resize(display, (int(dw*scale), int(dh*scale)))

            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                counter.calibrate(len(detections))
            elif key == ord('u'):
                counter.calibrated = False
                print("Unlocked!")
            elif key == ord('r'):
                counter.set_roi(frame)
            elif key == ord('+') or key == ord('='):
                counter.confidence = min(0.9, counter.confidence + 0.05)
                print(f"Conf: {counter.confidence:.2f}")
            elif key == ord('-'):
                counter.confidence = max(0.1, counter.confidence - 0.05)
                print(f"Conf: {counter.confidence:.2f}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("YOLO-World Counter - Fast text-based detection")
        print("Usage: python count_yoloworld.py <camera_ip>")
        print("Example: python count_yoloworld.py .129")
    else:
        run(sys.argv[1])
