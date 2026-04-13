#!/usr/bin/env python3
"""
Setup go2rtc for Tapo Two-Way Audio!
go2rtc supports Tapo camera speaker natively!
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import time


def download_go2rtc():
    """Download go2rtc for Windows."""
    print("\n[1] Downloading go2rtc...")

    go2rtc_dir = os.path.join(os.path.dirname(__file__), "go2rtc")
    os.makedirs(go2rtc_dir, exist_ok=True)

    exe_path = os.path.join(go2rtc_dir, "go2rtc.exe")

    if os.path.exists(exe_path):
        print(f"  Already exists: {exe_path}")
        return exe_path

    # Download latest release
    url = "https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_win64.zip"
    zip_path = os.path.join(go2rtc_dir, "go2rtc.zip")

    print(f"  Downloading from: {url}")
    urllib.request.urlretrieve(url, zip_path)

    # Extract
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(go2rtc_dir)

    os.remove(zip_path)
    print(f"  ✓ Installed: {exe_path}")

    return exe_path


def create_config(camera_ip, cloud_user, cloud_pass):
    """Create go2rtc config for Tapo camera."""
    print("\n[2] Creating config...")

    go2rtc_dir = os.path.join(os.path.dirname(__file__), "go2rtc")
    config_path = os.path.join(go2rtc_dir, "go2rtc.yaml")

    # For Tapo cameras, go2rtc uses special URL format
    # rtsp://user:pass@ip:554/stream1 for video+audio
    # With backchannel support!

    config = f"""
# go2rtc config for Tapo TC74
# Supports two-way audio!

streams:
  tapo_camera:
    - rtsp://admin:admin@{camera_ip}:554/stream1
    # Or use Tapo cloud format:
    # - tapo://{cloud_user}:{cloud_pass}@{camera_ip}

  tapo_hd:
    - rtsp://{cloud_user}:{cloud_pass}@{camera_ip}:554/stream1

  tapo_sd:
    - rtsp://{cloud_user}:{cloud_pass}@{camera_ip}:554/stream2

# Enable WebRTC for browser access with two-way audio
webrtc:
  listen: ":8555"

# API server
api:
  listen: ":1984"

# Enable RTSP server
rtsp:
  listen: ":8554"

# Log level
log:
  level: info
"""

    with open(config_path, 'w') as f:
        f.write(config)

    print(f"  ✓ Config saved: {config_path}")
    return config_path


def start_go2rtc():
    """Start go2rtc server."""
    print("\n[3] Starting go2rtc...")

    go2rtc_dir = os.path.join(os.path.dirname(__file__), "go2rtc")
    exe_path = os.path.join(go2rtc_dir, "go2rtc.exe")
    config_path = os.path.join(go2rtc_dir, "go2rtc.yaml")

    if not os.path.exists(exe_path):
        print("  go2rtc not found!")
        return None

    # Start go2rtc
    print(f"  Running: {exe_path}")
    process = subprocess.Popen(
        [exe_path, "-c", config_path],
        cwd=go2rtc_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Wait for startup
    time.sleep(2)

    if process.poll() is None:
        print("  ✓ go2rtc running!")
        print("\n  Access points:")
        print("    - Web UI: http://localhost:1984")
        print("    - WebRTC: http://localhost:1984/stream.html?src=tapo_camera")
        print("    - RTSP: rtsp://localhost:8554/tapo_camera")
        return process
    else:
        print("  Failed to start!")
        return None


def main():
    print("=" * 60)
    print("  GO2RTC SETUP FOR TAPO TWO-WAY AUDIO")
    print("=" * 60)
    print("""
  go2rtc is a streaming server that supports:
  - Tapo camera two-way audio!
  - WebRTC (browser access with mic)
  - Low latency streaming
    """)

    # Get camera info
    if len(sys.argv) >= 2:
        camera_ip = sys.argv[1]
    else:
        camera_ip = input("Camera IP (.129): ").strip()

    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"

    # Get credentials
    print("\nEnter Tapo credentials:")
    cloud_user = input("  Username (camera or cloud): ").strip() or "fasspay"
    cloud_pass = input("  Password: ").strip() or "fasspay2025"

    # Download go2rtc
    exe_path = download_go2rtc()
    if not exe_path:
        return

    # Create config
    config_path = create_config(camera_ip, cloud_user, cloud_pass)

    # Start server
    process = start_go2rtc()

    if process:
        print("\n" + "=" * 60)
        print("  GO2RTC READY!")
        print("=" * 60)
        print("""
  TO USE TWO-WAY AUDIO:

  1. Open browser: http://localhost:1984

  2. Click on your camera stream

  3. Click "WebRTC" mode

  4. Allow microphone access in browser

  5. Now you have TWO-WAY AUDIO!
     - Hear camera audio
     - Speak through camera speaker!

  Press Ctrl+C to stop go2rtc
        """)

        try:
            # Keep running
            while True:
                line = process.stdout.readline()
                if line:
                    print(f"  [go2rtc] {line.strip()}")
                if process.poll() is not None:
                    break
        except KeyboardInterrupt:
            print("\n  Stopping go2rtc...")
            process.terminate()


if __name__ == "__main__":
    main()
