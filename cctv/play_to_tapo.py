#!/usr/bin/env python3
"""
Play audio file to Tapo speaker via go2rtc
"""

import subprocess
import sys
import os
import time

CAMERA_IP = "192.168.122.129"
GO2RTC_API = "http://localhost:1984"

def play_audio(audio_path):
    """Send audio to Tapo via go2rtc."""

    print(f"Audio: {audio_path}")

    if not os.path.exists(audio_path):
        print("File not found!")
        return

    # Convert to format camera accepts (G.711 u-law, 8kHz mono)
    temp_file = "temp_audio.wav"

    print("Converting audio...")
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ar", "8000",
        "-ac", "1",
        "-acodec", "pcm_mulaw",
        "-t", "10",  # First 10 seconds only
        temp_file
    ]
    subprocess.run(cmd, capture_output=True)

    print("Sending to camera via go2rtc RTP...")

    # go2rtc accepts RTP input - send audio as RTP stream
    # go2rtc will forward to camera backchannel

    cmd = [
        "ffmpeg", "-re",
        "-i", temp_file,
        "-ar", "8000",
        "-ac", "1",
        "-c:a", "pcm_mulaw",
        "-f", "rtp",
        "rtp://127.0.0.1:5004"  # Send to local port, go2rtc can pick up
    ]

    print("Streaming...")
    try:
        subprocess.run(cmd, timeout=15)
    except:
        pass

    # Cleanup
    try:
        os.remove(temp_file)
    except:
        pass

    print("Done!")


if __name__ == "__main__":
    audio_path = r"C:\Users\Cheng Jit Giam\Downloads\Mambu OMatsuri mambu - (320 Kbps).mp3"

    print("=" * 50)
    print("  PLAY AUDIO TO TAPO SPEAKER")
    print("=" * 50)

    # Check go2rtc
    import requests
    try:
        resp = requests.get(f"{GO2RTC_API}/api/streams", timeout=2)
        print(f"go2rtc running: {list(resp.json().keys())}")
    except:
        print("go2rtc not running! Start it first.")
        sys.exit(1)

    play_audio(audio_path)
