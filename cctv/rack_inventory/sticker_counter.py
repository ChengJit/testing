#!/usr/bin/env python3
"""
Sticker-Based Box Counter
==========================

Counts boxes by detecting WHITE STICKERS on each box.
Much more accurate than detecting box edges!

Each sticker = 1 box
"""

import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


@dataclass
class StickerDetection:
    """Detected sticker."""
    x: int  # center x
    y: int  # center y
    w: int  # width
    h: int  # height
    area: int


class StickerCounter:
    """
    Count boxes by detecting white stickers.
    """

    def __init__(
        self,
        # White color range in HSV
        # Adjust these if stickers aren't being detected
        min_saturation: int = 0,
        max_saturation: int = 50,  # Low saturation = white
        min_value: int = 180,       # High brightness = white
        # Sticker size range (pixels)
        min_area: int = 100,
        max_area: int = 5000,
        min_width: int = 10,
        max_width: int = 100,
        # Shape constraints
        min_aspect: float = 0.3,  # width/height ratio
        max_aspect: float = 3.0,
    ):
        self.min_sat = min_saturation
        self.max_sat = max_saturation
        self.min_val = min_value
        self.min_area = min_area
        self.max_area = max_area
        self.min_width = min_width
        self.max_width = max_width
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

        # ROI
        self.roi: Optional[Tuple[float, float, float, float]] = None

        # State
        self.last_count = 0
        self.baseline = 0

    def detect_stickers(self, frame: np.ndarray) -> List[StickerDetection]:
        """Detect white stickers in frame."""
        h, w = frame.shape[:2]

        # Apply ROI
        if self.roi:
            x1 = int(self.roi[0] * w)
            y1 = int(self.roi[1] * h)
            x2 = int(self.roi[2] * w)
            y2 = int(self.roi[3] * h)
            work_frame = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            work_frame = frame
            offset = (0, 0)

        # Convert to HSV
        hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)

        # White color mask (any hue, low saturation, high value)
        lower_white = np.array([0, self.min_sat, self.min_val])
        upper_white = np.array([180, self.max_sat, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stickers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Get bounding rectangle
            x, y, sw, sh = cv2.boundingRect(cnt)

            # Filter by size
            if sw < self.min_width or sw > self.max_width:
                continue

            # Filter by aspect ratio
            aspect = sw / sh if sh > 0 else 0
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            # Calculate center
            cx = x + sw // 2 + offset[0]
            cy = y + sh // 2 + offset[1]

            stickers.append(StickerDetection(
                x=cx, y=cy, w=sw, h=sh, area=area
            ))

        # Remove duplicates (stickers too close together)
        stickers = self._remove_duplicates(stickers)

        return stickers

    def _remove_duplicates(self, stickers: List[StickerDetection], min_dist: int = 30) -> List[StickerDetection]:
        """Remove stickers that are too close together."""
        if not stickers:
            return []

        # Sort by area (keep larger ones)
        stickers = sorted(stickers, key=lambda s: s.area, reverse=True)

        kept = []
        for sticker in stickers:
            is_duplicate = False
            for existing in kept:
                dist = ((sticker.x - existing.x) ** 2 + (sticker.y - existing.y) ** 2) ** 0.5
                if dist < min_dist:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(sticker)

        return kept

    def count(self, frame: np.ndarray) -> Tuple[int, List[StickerDetection]]:
        """Count stickers (boxes)."""
        stickers = self.detect_stickers(frame)
        self.last_count = len(stickers)
        return self.last_count, stickers

    def set_roi_interactive(self, frame: np.ndarray):
        """Let user draw ROI."""
        print("Draw ROI around the box area")
        print("  Click and drag, then press ENTER")

        roi_pts = []
        drawing = False
        temp = frame.copy()

        def mouse_cb(event, x, y, flags, param):
            nonlocal roi_pts, drawing, temp
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_pts = [(x, y)]
                drawing = True
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp = frame.copy()
                cv2.rectangle(temp, roi_pts[0], (x, y), (0, 255, 255), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                roi_pts.append((x, y))
                drawing = False

        win = "Draw ROI - ENTER to confirm"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, mouse_cb)

        while True:
            cv2.imshow(win, temp)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and len(roi_pts) == 2:
                break
            elif key == 27:
                cv2.destroyWindow(win)
                return

        cv2.destroyWindow(win)

        h, w = frame.shape[:2]
        x1 = min(roi_pts[0][0], roi_pts[1][0]) / w
        y1 = min(roi_pts[0][1], roi_pts[1][1]) / h
        x2 = max(roi_pts[0][0], roi_pts[1][0]) / w
        y2 = max(roi_pts[0][1], roi_pts[1][1]) / h

        self.roi = (x1, y1, x2, y2)
        print(f"ROI set: {self.roi}")

    def draw_overlay(self, frame: np.ndarray, stickers: List[StickerDetection]) -> np.ndarray:
        """Draw detection overlay."""
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
        for i, sticker in enumerate(stickers):
            # Circle around sticker
            cv2.circle(display, (sticker.x, sticker.y), 15, (0, 255, 0), 2)
            # Number
            cv2.putText(display, str(i + 1), (sticker.x - 5, sticker.y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Count panel
        count = len(stickers)
        change = count - self.baseline if self.baseline > 0 else 0

        cv2.rectangle(display, (10, 10), (250, 90), (0, 0, 0), -1)
        cv2.putText(display, f"Stickers: {count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.baseline > 0:
            change_color = (0, 255, 0) if change >= 0 else (0, 0, 255)
            cv2.putText(display, f"Baseline: {self.baseline}", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(display, f"Change: {change:+d}", (150, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, change_color, 1)

        return display

    def show_debug(self, frame: np.ndarray) -> np.ndarray:
        """Show debug view with mask."""
        h, w = frame.shape[:2]

        # Apply ROI
        if self.roi:
            x1 = int(self.roi[0] * w)
            y1 = int(self.roi[1] * h)
            x2 = int(self.roi[2] * w)
            y2 = int(self.roi[3] * h)
            work_frame = frame[y1:y2, x1:x2]
        else:
            work_frame = frame

        # Convert to HSV
        hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)

        # White mask
        lower_white = np.array([0, self.min_sat, self.min_val])
        upper_white = np.array([180, self.max_sat, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Convert mask to 3 channel for display
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Resize to match
        if mask_color.shape[:2] != work_frame.shape[:2]:
            mask_color = cv2.resize(mask_color, (work_frame.shape[1], work_frame.shape[0]))

        # Stack side by side
        debug = np.hstack([work_frame, mask_color])

        # Labels
        cv2.putText(debug, "Camera", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug, "White Mask", (work_frame.shape[1] + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return debug


def run_sticker_counter(camera_ip: str):
    """Run sticker counter."""
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2?rtsp_transport=tcp"

    print(f"Connecting to {camera_ip}...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Failed to connect!")
        return

    print("Connected!")

    counter = StickerCounter()

    # Get first frame
    cap.grab()
    ret, frame = cap.retrieve()
    if ret:
        counter.set_roi_interactive(frame)

    print("\nControls:")
    print("  B - Set BASELINE")
    print("  R - Redraw ROI")
    print("  D - Toggle DEBUG view (show white mask)")
    print("  +/- - Adjust white sensitivity")
    print("  S - Screenshot")
    print("  Q - Quit")
    print("-" * 50)

    window = "Sticker Counter (1 sticker = 1 box)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    debug_mode = False
    last_detect = 0
    stickers = []

    try:
        while True:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                continue

            # Detect every 1 second
            if time.time() - last_detect >= 1.0:
                count, stickers = counter.count(frame)
                last_detect = time.time()

                if counter.baseline > 0:
                    change = count - counter.baseline
                    if change != 0:
                        print(f"[{datetime.now():%H:%M:%S}] Count: {count} (change: {change:+d})")

            # Display
            if debug_mode:
                display = counter.show_debug(frame)
            else:
                display = counter.draw_overlay(frame, stickers)

            # Instructions
            h = display.shape[0]
            cv2.rectangle(display, (0, h - 30), (display.shape[1], h), (40, 40, 40), -1)
            mode = "DEBUG" if debug_mode else "NORMAL"
            cv2.putText(display,
                       f"[{mode}] B:Baseline R:ROI D:Debug +/-:Sensitivity Q:Quit",
                       (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow(window, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                counter.baseline = counter.last_count
                print(f"Baseline set: {counter.baseline}")
            elif key == ord('r'):
                counter.set_roi_interactive(frame)
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug mode: {debug_mode}")
            elif key == ord('+') or key == ord('='):
                counter.min_val = max(100, counter.min_val - 10)
                print(f"White threshold: {counter.min_val} (more sensitive)")
            elif key == ord('-'):
                counter.min_val = min(250, counter.min_val + 10)
                print(f"White threshold: {counter.min_val} (less sensitive)")
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
        print("Usage: python sticker_counter.py <camera_ip>")
        print("Example: python sticker_counter.py .129")
    else:
        run_sticker_counter(sys.argv[1])
