#!/usr/bin/env python3
"""
Simple camera preview - no detection, just view.
Press S to save screenshot.
"""

import cv2
import numpy as np
import sys
import os
import time

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


def preview_cameras(camera_ips):
    """Show camera feeds side by side."""

    # Build RTSP URLs
    cameras = []
    for ip in camera_ips:
        if ip.startswith('.'):
            ip = f"192.168.122{ip}"
        url = f"rtsp://fasspay:fasspay2025@{ip}:554/stream2?rtsp_transport=tcp"
        cameras.append({"ip": ip, "url": url, "cap": None})

    print(f"Connecting to {len(cameras)} camera(s)...")

    # Connect
    for cam in cameras:
        cap = cv2.VideoCapture(cam["url"], cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam["cap"] = cap
        if cap.isOpened():
            print(f"  [OK] {cam['ip']}")
        else:
            print(f"  [FAIL] {cam['ip']}")

    print("\nControls:")
    print("  S - Save screenshot")
    print("  Q - Quit")
    print("-" * 40)

    window_name = "Camera Preview - Press S to screenshot, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = []

            for cam in cameras:
                cap = cam["cap"]
                if cap and cap.isOpened():
                    cap.grab()
                    ret, frame = cap.retrieve()
                    if ret:
                        # Resize for display
                        h, w = frame.shape[:2]
                        if w > 800:
                            scale = 800 / w
                            frame = cv2.resize(frame, (800, int(h * scale)))

                        # Add camera label
                        cv2.rectangle(frame, (0, 0), (200, 35), (0, 0, 0), -1)
                        cv2.putText(frame, cam["ip"], (10, 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        frames.append(frame)
                    else:
                        frames.append(np.zeros((450, 800, 3), dtype=np.uint8))
                else:
                    placeholder = np.zeros((450, 800, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"{cam['ip']} - Disconnected", (50, 225),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    frames.append(placeholder)

            # Stack frames
            if len(frames) == 1:
                display = frames[0]
            elif len(frames) == 2:
                # Resize to same height
                h = max(f.shape[0] for f in frames)
                resized = []
                for f in frames:
                    if f.shape[0] != h:
                        scale = h / f.shape[0]
                        f = cv2.resize(f, (int(f.shape[1] * scale), h))
                    resized.append(f)
                display = np.hstack(resized)
            else:
                display = frames[0]

            # Add instructions
            h = display.shape[0]
            cv2.rectangle(display, (0, h-30), (display.shape[1], h), (40, 40, 40), -1)
            cv2.putText(display, "S: Screenshot | Q: Quit", (10, h-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"rack_preview_{int(time.time())}.jpg"
                cv2.imwrite(filename, display)
                print(f"Saved: {filename}")

    finally:
        for cam in cameras:
            if cam["cap"]:
                cam["cap"].release()
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Usage: python preview.py <camera_ips>")
        print("Example: python preview.py .128 .129")
        return

    camera_ips = sys.argv[1:]
    preview_cameras(camera_ips)


if __name__ == "__main__":
    main()
