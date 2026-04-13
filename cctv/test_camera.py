#!/usr/bin/env python3
"""
Quick camera test - low latency single camera view.
Usage: python test_camera.py 192.168.122.128
"""

import sys
import cv2
import time

# Default credentials
USER = "fasspay"
PASS = "fasspay2025"


def test_camera(ip: str):
    """Test a single camera with minimal latency."""

    # Use SD stream (stream2) for lower latency
    url = f"rtsp://{USER}:{PASS}@{ip}:554/stream2?rtsp_transport=tcp"

    print(f"Connecting to {ip}...")
    print(f"URL: {url}")
    print()

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: Cannot connect to {ip}")
        print("Check:")
        print("  - Camera is powered on")
        print("  - IP address is correct")
        print("  - RTSP is enabled in Tapo app")
        print("  - Credentials: fasspay:fasspay2025")
        return

    print(f"Connected! Press Q to quit.")
    print("-" * 40)

    frame_count = 0
    start_time = time.time()

    while True:
        # Grab and retrieve (skip buffered frames)
        cap.grab()
        ret, frame = cap.retrieve()

        if not ret:
            print("Lost connection!")
            break

        frame_count += 1
        elapsed = time.time() - start_time

        # Calculate FPS every 30 frames
        if frame_count % 30 == 0:
            fps = frame_count / elapsed
            # Add FPS overlay
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize for display (smaller = faster)
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        cv2.imshow(f"Camera: {ip}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    fps = frame_count / (time.time() - start_time)
    print(f"\nAverage FPS: {fps:.1f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_camera.py <IP>")
        print("Example: python test_camera.py 192.168.122.128")
        print()

        # Try to scan for cameras
        print("Scanning for cameras...")
        for i in range(125, 135):
            ip = f"192.168.122.{i}"
            url = f"rtsp://{USER}:{PASS}@{ip}:554/stream2?rtsp_transport=tcp"
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"  FOUND: {ip}")
            else:
                cap.release()
    else:
        test_camera(sys.argv[1])
