#!/usr/bin/env python3
"""
Send audio to Tapo speaker via go2rtc!
"""

import os
import sys
import time
import requests
import subprocess
import tempfile


def create_audio(text="Hi! How can I help?"):
    """Create audio file."""
    audio_file = os.path.join(os.path.dirname(__file__), "speak.wav")

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file(text, audio_file)
        engine.runAndWait()
        engine.stop()
        time.sleep(0.3)
        print(f"Created audio: {audio_file}")
        return audio_file
    except Exception as e:
        print(f"Error: {e}")
        return None


def send_via_ffmpeg(audio_file, stream_name="tapo_camera"):
    """Send audio to go2rtc via RTSP."""
    print("\n[Method 1] Sending via FFmpeg to go2rtc...")

    # go2rtc RTSP is on port 8554
    rtsp_url = f"rtsp://localhost:8554/{stream_name}"

    # Convert and send
    cmd = [
        "ffmpeg", "-re",
        "-i", audio_file,
        "-ar", "8000",
        "-ac", "1",
        "-c:a", "pcm_mulaw",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url
    ]

    print(f"  Sending to: {rtsp_url}")
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10, text=True)
        print(f"  Done!")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def send_via_api(audio_file, stream_name="tapo_camera"):
    """Send audio via go2rtc API."""
    print("\n[Method 2] Sending via go2rtc API...")

    api_url = f"http://localhost:1984/api/streams/{stream_name}"

    # Check if stream exists
    try:
        resp = requests.get(f"http://localhost:1984/api/streams")
        print(f"  Streams: {resp.json()}")
    except Exception as e:
        print(f"  API error: {e}")
        return False

    return False


def send_via_webrtc_backchannel():
    """Instructions for WebRTC backchannel."""
    print("\n[Method 3] WebRTC Backchannel (Browser)...")
    print("""
  For two-way audio via browser:

  1. Open: http://localhost:1984
  2. Click on 'tapo_camera' or 'tapo_hd'
  3. Select 'webrtc' as the player
  4. Allow microphone in browser
  5. Speak - audio goes to camera speaker!
    """)


def send_direct_to_camera(audio_file, camera_ip):
    """Send directly to camera using go2rtc's backchannel."""
    print("\n[Method 4] Direct backchannel via go2rtc...")

    # go2rtc should proxy the backchannel
    # Try sending to go2rtc's RTSP which forwards to camera

    # First check go2rtc streams
    try:
        resp = requests.get("http://localhost:1984/api/streams")
        streams = resp.json()
        print(f"  Available streams: {list(streams.keys())}")

        for name, info in streams.items():
            print(f"\n  Stream: {name}")
            print(f"    Producers: {info.get('producers', [])}")
            print(f"    Consumers: {info.get('consumers', [])}")

    except Exception as e:
        print(f"  Error: {e}")

    return False


def play_test():
    """Quick test - play audio locally first."""
    print("\n[Test] Playing audio locally...")

    audio_file = create_audio("Hi! How can I help?")
    if not audio_file:
        return

    if sys.platform == "win32":
        subprocess.run(
            ['powershell', '-c', f'(New-Object Media.SoundPlayer "{audio_file}").PlaySync()'],
            capture_output=True
        )
        print("  Played on laptop speaker!")


def main():
    print("=" * 60)
    print("  SEND AUDIO TO TAPO VIA GO2RTC")
    print("=" * 60)

    # Check if go2rtc is running
    try:
        resp = requests.get("http://localhost:1984/api/streams", timeout=2)
        print("\n✓ go2rtc is running!")
    except:
        print("\n✗ go2rtc not running!")
        print("  Start it first: python setup_go2rtc.py .129")
        return

    # Create audio
    audio_file = create_audio("Hi! How can I help?")

    # Get camera IP
    camera_ip = "192.168.122.129"

    # Try methods
    send_via_ffmpeg(audio_file, "tapo_hd")
    send_via_api(audio_file, "tapo_hd")
    send_direct_to_camera(audio_file, camera_ip)
    send_via_webrtc_backchannel()

    print("\n" + "=" * 60)
    print("  BEST WAY TO TEST TWO-WAY AUDIO:")
    print("=" * 60)
    print("""
  1. Open browser: http://localhost:1984

  2. Click 'add' to add stream if not there:
     - Name: tapo
     - URL: rtsp://fasspay:fasspay2025@192.168.122.129:554/stream1

  3. Click the stream, then click 'stream'

  4. Select 'webrtc' player

  5. Click 'microphone' icon to enable

  6. SPEAK! Your voice goes to camera speaker!

  This is the easiest way for two-way audio!
    """)


if __name__ == "__main__":
    main()
