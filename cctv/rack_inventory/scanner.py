"""
Camera Scanner - Find Tapo cameras on the network.
"""

import cv2
import concurrent.futures
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FoundCamera:
    """Represents a discovered camera."""
    ip: str
    rtsp_url: str
    name: str = ""


def test_camera(ip: str, username: str, password: str, timeout: int = 5) -> Optional[FoundCamera]:
    """Test if RTSP camera is accessible at given IP."""
    # Use stream2 (SD) for faster connection test
    url = f"rtsp://{username}:{password}@{ip}:554/stream2?rtsp_transport=tcp"

    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return FoundCamera(ip=ip, rtsp_url=url)
        cap.release()
    except Exception as e:
        pass

    return None


def scan_network(
    network_prefix: str = "192.168.122",
    ip_range: range = range(1, 255),
    username: str = "fasspay",
    password: str = "fasspay2025",
    max_workers: int = 20,
    progress_callback=None,
) -> List[FoundCamera]:
    """
    Scan network for RTSP cameras.

    Args:
        network_prefix: First 3 octets of IP (e.g., "192.168.122")
        ip_range: Range of last octet to scan
        username: RTSP username
        password: RTSP password
        max_workers: Parallel scan threads
        progress_callback: Optional callback(current, total, ip, found)

    Returns:
        List of found cameras
    """
    found_cameras = []
    ips_to_scan = [f"{network_prefix}.{i}" for i in ip_range]
    total = len(ips_to_scan)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(test_camera, ip, username, password): ip
            for ip in ips_to_scan
        }

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            ip = futures[future]
            result = future.result()

            if progress_callback:
                progress_callback(i + 1, total, ip, result is not None)

            if result:
                found_cameras.append(result)

    # Sort by IP
    found_cameras.sort(key=lambda c: [int(x) for x in c.ip.split('.')])

    return found_cameras


def quick_scan(
    ips: List[str],
    username: str = "fasspay",
    password: str = "fasspay2025",
) -> List[FoundCamera]:
    """Quick scan of specific IPs."""
    found = []
    for ip in ips:
        print(f"Testing {ip}...", end=" ")
        result = test_camera(ip, username, password)
        if result:
            print("FOUND!")
            found.append(result)
        else:
            print("not found")
    return found


if __name__ == "__main__":
    print("=" * 50)
    print("  Tapo Camera Scanner")
    print("=" * 50)
    print()

    # Quick scan common IPs first
    print("Quick scan of likely IPs...")
    print("-" * 30)

    likely_ips = [f"192.168.122.{i}" for i in range(125, 135)]
    cameras = quick_scan(likely_ips)

    if cameras:
        print(f"\nFound {len(cameras)} camera(s):")
        for i, cam in enumerate(cameras, 1):
            print(f"  {i}. {cam.ip}")
            print(f"     RTSP: {cam.rtsp_url}")
    else:
        print("\nNo cameras found in quick scan.")
        print("Running full network scan...")

        def progress(current, total, ip, found):
            if current % 25 == 0:
                print(f"  Scanned {current}/{total}...")

        cameras = scan_network(progress_callback=progress)

        if cameras:
            print(f"\nFound {len(cameras)} camera(s):")
            for i, cam in enumerate(cameras, 1):
                print(f"  {i}. {cam.ip}")
        else:
            print("\nNo cameras found. Check:")
            print("  - Cameras are powered on")
            print("  - Cameras are on 192.168.122.x network")
            print("  - RTSP is enabled in Tapo app")
            print("  - Credentials are correct (fasspay:fasspay2025)")
