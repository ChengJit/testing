#!/usr/bin/env python3
"""
Simple: Send audio file to Tapo speaker via go2rtc
"""

import os
import subprocess
import time

# Camera settings
CAMERA_IP = "192.168.122.129"
USERNAME = "fasspay"
PASSWORD = "fasspay2025"


def create_audio():
    """Create 'Hi how can I help' audio."""
    audio_file = os.path.join(os.path.dirname(__file__), "hi_help.wav")

    if os.path.exists(audio_file):
        return audio_file

    print("Creating audio...")
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.save_to_file("Hi! How can I help?", audio_file)
    engine.runAndWait()
    engine.stop()
    time.sleep(0.5)
    print(f"Created: {audio_file}")
    return audio_file


def send_audio(audio_file):
    """Send audio to Tapo speaker via go2rtc backchannel."""
    print(f"\nSending audio to Tapo speaker...")

    # go2rtc exposes backchannel on the RTSP stream
    # We send audio as RTP to go2rtc which forwards to camera

    # Method: FFmpeg to go2rtc's RTSP input
    rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/stream1"

    cmd = [
        "ffmpeg",
        "-re",                          # Real-time
        "-i", audio_file,               # Input audio
        "-ar", "8000",                  # Sample rate
        "-ac", "1",                     # Mono
        "-acodec", "pcm_alaw",          # G.711 A-law (common for IP cameras)
        "-f", "rtp",                    # Output format
        "-sdp_file", "audio.sdp",       # Create SDP file
        f"rtp://{CAMERA_IP}:5000"       # Send to camera
    ]

    print(f"Running: ffmpeg -> camera")

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10, text=True)
        print("Sent!")
    except subprocess.TimeoutExpired:
        print("Sent (timeout ok)")
    except Exception as e:
        print(f"Error: {e}")


def send_via_go2rtc(audio_file):
    """Send via go2rtc backchannel API."""
    print(f"\nSending via go2rtc backchannel...")

    # go2rtc has a special backchannel endpoint
    # POST audio to: http://localhost:1984/api/streams/tapo_hd/backchannel

    import requests

    # First, start backchannel session
    try:
        # Check if stream exists
        resp = requests.get("http://localhost:1984/api/streams")
        streams = resp.json()
        print(f"Streams: {list(streams.keys())}")

        # Read audio file
        with open(audio_file, 'rb') as f:
            audio_data = f.read()

        # Try to send to backchannel
        for stream_name in streams.keys():
            url = f"http://localhost:1984/api/streams/{stream_name}/send"
            print(f"Trying: {url}")

            try:
                resp = requests.post(url, data=audio_data,
                                   headers={'Content-Type': 'audio/wav'},
                                   timeout=5)
                print(f"  Response: {resp.status_code}")
            except:
                pass

    except Exception as e:
        print(f"Error: {e}")


def send_direct_rtsp(audio_file):
    """Send audio directly to camera RTSP backchannel."""
    print(f"\nSending direct to camera RTSP backchannel...")

    # Tapo cameras may accept audio on backchannel via RTSP ANNOUNCE
    rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/stream1"

    cmd = [
        "ffmpeg",
        "-re",
        "-i", audio_file,
        "-ar", "8000",
        "-ac", "1",
        "-c:a", "pcm_mulaw",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url
    ]

    print(f"Sending to: {rtsp_url}")

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10, text=True)
        if "200" in result.stderr or result.returncode == 0:
            print("Success!")
            return True
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        print(f"Error: {e}")

    return False


if __name__ == "__main__":
    print("=" * 50)
    print("  SEND AUDIO TO TAPO SPEAKER")
    print("=" * 50)

    # Create audio
    audio_file = create_audio()

    # Try sending
    send_via_go2rtc(audio_file)
    send_direct_rtsp(audio_file)
    send_audio(audio_file)

    print("\n" + "=" * 50)
    print("Did you hear 'Hi how can I help?' from camera?")
    print("=" * 50)
