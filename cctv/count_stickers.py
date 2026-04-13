#!/usr/bin/env python3
"""
Fast Sticker Counter - No AI needed!
Detect white stickers on boxes using simple color detection.
"""

import cv2
import numpy as np
import time
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


class StickerCounter:
    def __init__(self):
        self.roi = None
        self.baseline = 0
        self.last_count = 0

        # White sticker detection settings
        self.min_area = 200      # Min sticker size
        self.max_area = 8000     # Max sticker size
        self.brightness = 200    # Min brightness for "white"

    def detect_stickers(self, frame):
        """Find white stickers in frame."""
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

        # Convert to grayscale
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

        # Find bright (white) regions
        _, thresh = cv2.threshold(gray, self.brightness, 255, cv2.THRESH_BINARY)

        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stickers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Filter by aspect ratio (stickers roughly square-ish)
                aspect = bw / bh if bh > 0 else 0
                if 0.3 < aspect < 3.0:
                    stickers.append({
                        'x': x + offset[0],
                        'y': y + offset[1],
                        'w': bw,
                        'h': bh,
                        'area': area
                    })

        self.last_count = len(stickers)
        return stickers

    def set_roi(self, frame):
        """Draw ROI interactively."""
        pts = []
        drawing = False
        temp = frame.copy()

        def mouse(event, x, y, flags, param):
            nonlocal pts, drawing, temp
            if event == cv2.EVENT_LBUTTONDOWN:
                pts = [(x, y)]
                drawing = True
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp = frame.copy()
                cv2.rectangle(temp, pts[0], (x, y), (0, 255, 255), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                pts.append((x, y))
                drawing = False

        win = "Draw ROI - ENTER to confirm, ESC to skip"
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

        h, w = frame.shape[:2]
        self.roi = (
            min(pts[0][0], pts[1][0]) / w,
            min(pts[0][1], pts[1][1]) / h,
            max(pts[0][0], pts[1][0]) / w,
            max(pts[0][1], pts[1][1]) / h,
        )
        print(f"ROI set!")

    def draw(self, frame, stickers):
        """Draw results on frame."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw ROI
        if self.roi:
            rx1 = int(self.roi[0] * w)
            ry1 = int(self.roi[1] * h)
            rx2 = int(self.roi[2] * w)
            ry2 = int(self.roi[3] * h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

        # Draw stickers
        for i, s in enumerate(stickers):
            cv2.rectangle(display, (s['x'], s['y']),
                         (s['x'] + s['w'], s['y'] + s['h']), (0, 255, 0), 2)
            cv2.putText(display, str(i + 1), (s['x'] + 5, s['y'] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Count box
        cv2.rectangle(display, (10, 10), (200, 90), (0, 0, 0), -1)
        cv2.putText(display, f"Boxes: {len(stickers)}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        if self.baseline > 0:
            change = len(stickers) - self.baseline
            color = (0, 255, 0) if change >= 0 else (0, 0, 255)
            cv2.putText(display, f"Change: {change:+d}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Help bar
        cv2.rectangle(display, (0, h - 25), (w, h), (40, 40, 40), -1)
        cv2.putText(display, "B:Baseline R:ROI +/-:Brightness S:Save Q:Quit",
                   (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Settings
        cv2.putText(display, f"Bright:{self.brightness}", (w - 100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return display


def run(camera_ip):
    """Run sticker counter."""
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2?rtsp_transport=tcp"

    print("=" * 40)
    print("  STICKER COUNTER - Fast & Simple")
    print("=" * 40)
    print(f"\nConnecting to {camera_ip}...")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("No connect! Check camera.")
        return

    print("Connected!")

    counter = StickerCounter()

    # Get first frame for ROI
    cap.grab()
    ret, frame = cap.retrieve()
    if ret:
        counter.set_roi(frame)

    print("\nControls:")
    print("  B - Set baseline")
    print("  R - Redraw ROI")
    print("  +/- - Adjust brightness threshold")
    print("  S - Screenshot")
    print("  Q - Quit")
    print("-" * 40)

    win = "Sticker Counter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                continue

            # Detect (instant - no AI!)
            stickers = counter.detect_stickers(frame)

            # Draw
            display = counter.draw(frame, stickers)
            cv2.imshow(win, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                counter.baseline = counter.last_count
                print(f"Baseline: {counter.baseline}")
            elif key == ord('r'):
                counter.set_roi(frame)
            elif key == ord('+') or key == ord('='):
                counter.brightness = min(255, counter.brightness + 10)
                print(f"Brightness: {counter.brightness}")
            elif key == ord('-'):
                counter.brightness = max(100, counter.brightness - 10)
                print(f"Brightness: {counter.brightness}")
            elif key == ord('s'):
                fname = f"sticker_{int(time.time())}.jpg"
                cv2.imwrite(fname, display)
                print(f"Saved: {fname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Sticker Counter")
        print("=" * 30)
        print()
        print("Fast box counting using white stickers.")
        print("No AI - instant detection!")
        print()
        print("Usage: python count_stickers.py <camera_ip>")
        print("Example: python count_stickers.py .129")
    else:
        run(sys.argv[1])
