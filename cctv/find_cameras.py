#!/usr/bin/env python3
"""
Quick scanner to find Tapo cameras on the network.
Tests RTSP connectivity on common IPs.
"""

import cv2
import concurrent.futures
from typing import List, Tuple

# Camera credentials (same for all Tapo cameras)
USERNAME = "fasspay"
PASSWORD = "fasspay2025"

# Network range to scan (adjust if needed)
NETWORK_PREFIX = "192.168.122"
IP_RANGE = range(1, 255)  # Scan .1 to .254

# Known camera (door monitor)
KNOWN_CAMERAS = {
    "192.168.122.127": "Door Camera (existing)"
}


def test_rtsp(ip: str, timeout: int = 3) -> Tuple[str, bool, str]:
    """Test if RTSP stream is accessible on given IP."""
    url = f"rtsp://{USERNAME}:{PASSWORD}@{ip}:554/stream1"

    try:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)

        if cap.isOpened():
            # Try to read a frame to confirm
            ret, _ = cap.read()
            cap.release()
            if ret:
                return ip, True, url
        cap.release()
    except Exception as e:
        pass

    return ip, False, ""


def scan_network() -> List[Tuple[str, str]]:
    """Scan network for RTSP cameras."""
    print(f"Scanning {NETWORK_PREFIX}.0/24 for Tapo cameras...")
    print(f"Using credentials: {USERNAME}:{'*' * len(PASSWORD)}")
    print("-" * 50)

    found_cameras = []
    ips_to_scan = [f"{NETWORK_PREFIX}.{i}" for i in IP_RANGE]

    # Use thread pool for parallel scanning
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(test_rtsp, ip): ip for ip in ips_to_scan}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            ip, success, url = future.result()

            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Scanned {i + 1}/{len(ips_to_scan)} IPs...")

            if success:
                label = KNOWN_CAMERAS.get(ip, "NEW CAMERA")
                print(f"  [FOUND] {ip} - {label}")
                found_cameras.append((ip, url))

    return found_cameras


def main():
    print("=" * 50)
    print("  Tapo Camera Scanner")
    print("=" * 50)
    print()

    cameras = scan_network()

    print()
    print("=" * 50)
    print(f"  Found {len(cameras)} camera(s)")
    print("=" * 50)

    for ip, url in sorted(cameras):
        label = KNOWN_CAMERAS.get(ip, "Stock Camera")
        print(f"\n{label}:")
        print(f"  IP:   {ip}")
        print(f"  RTSP: {url}")

    if len(cameras) > 1:
        print("\n" + "-" * 50)
        print("Add these to config.json or use multi-camera setup.")

        # Generate config snippet
        print("\nConfig snippet for new cameras:")
        for i, (ip, url) in enumerate(sorted(cameras)):
            if ip not in KNOWN_CAMERAS:
                print(f'''
  "camera_stock_{i+1}": {{
    "source": "{url}",
    "camera_id": "stock-cam-{i+1:03d}"
  }}''')


if __name__ == "__main__":
    main()
