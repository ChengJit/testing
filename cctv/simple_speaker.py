#!/usr/bin/env python3
"""
Simple: Play audio to Tapo speaker via FFmpeg + go2rtc
"""

import subprocess
import sys
import os
import time
import requests

GO2RTC_API = "http://localhost:1984"

def get_stream():
    """Get first available stream from go2rtc."""
    try:
        resp = requests.get(f"{GO2RTC_API}/api/streams", timeout=2)
        streams = list(resp.json().keys())
        print(f"Available streams: {streams}")
        return streams[0] if streams else None
    except:
        return None

def play_to_speaker(audio_path, stream_name):
    """Play audio file to Tapo speaker via go2rtc RTSP."""

    print(f"\nPlaying: {audio_path}")
    print(f"To stream: {stream_name}")

    # go2rtc RTSP endpoint with backchannel
    rtsp_url = f"rtsp://localhost:8554/{stream_name}?backchannel=1"

    # FFmpeg: convert audio and send to go2rtc RTSP
    # go2rtc will forward to camera backchannel
    cmd = [
        "ffmpeg",
        "-re",                      # Real-time playback
        "-i", audio_path,           # Input file
        "-ar", "8000",              # 8kHz sample rate (G.711)
        "-ac", "1",                 # Mono
        "-c:a", "pcm_mulaw",        # G.711 μ-law codec
        "-f", "rtsp",               # Output format
        "-rtsp_transport", "tcp",   # Use TCP
        rtsp_url
    ]

    print(f"\nSending audio via RTSP...")
    print(f"Target: {rtsp_url}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("Done!")
        else:
            print(f"FFmpeg stderr: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print("Timeout (probably ok)")
    except Exception as e:
        print(f"Error: {e}")


def play_via_rtp(audio_path):
    """Alternative: Send via RTP directly."""

    print(f"\n[Method 2] Trying RTP...")

    # Some go2rtc setups accept RTP input
    cmd = [
        "ffmpeg",
        "-re",
        "-i", audio_path,
        "-ar", "8000",
        "-ac", "1",
        "-c:a", "pcm_mulaw",
        "-f", "rtp",
        "rtp://127.0.0.1:8555"  # go2rtc WebRTC port
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=15)
    except:
        pass


def create_test_audio():
    """Create a simple test beep."""
    test_file = "test_beep.wav"

    if os.path.exists(test_file):
        return test_file

    print("Creating test beep...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=2",  # 1kHz beep for 2 seconds
        "-ar", "8000",
        "-ac", "1",
        test_file
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"Created: {test_file}")
    return test_file


def main():
    print("=" * 50)
    print("  SIMPLE TAPO SPEAKER TEST")
    print("=" * 50)

    # Check go2rtc
    stream = get_stream()
    if not stream:
        print("\ngo2rtc not running!")
        print("Start it: cd go2rtc && .\\go2rtc.exe -c go2rtc.yaml")
        return

    # Get audio file
    if len(sys.argv) >= 2:
        audio_path = sys.argv[1]
    else:
        # Create test beep
        audio_path = create_test_audio()

    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return

    # Try playing
    play_to_speaker(audio_path, stream)

    print("\n" + "=" * 50)
    print("  Did you hear audio from the Tapo speaker?")
    print("=" * 50)
    print("""
If not, try the browser test:
  1. Open: http://localhost:1984
  2. Click stream -> webrtc player
  3. Enable microphone
  4. Speak - should come out camera speaker!
    """)


if __name__ == "__main__":
    main()
