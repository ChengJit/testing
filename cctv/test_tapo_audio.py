#!/usr/bin/env python3
"""
Test Tapo TC74 Audio - Check if we can hear mic!
"""

import subprocess
import sys
import os

def test_audio_stream(camera_ip):
    """Test if we can get audio from Tapo camera."""
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"

    # RTSP URL with audio
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream1"

    print("=" * 50)
    print("  TAPO AUDIO TEST")
    print("=" * 50)
    print(f"\nCamera: {camera_ip}")
    print(f"URL: {url}")

    # Method 1: Check stream info with ffprobe
    print("\n[1] Checking stream info...")
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", url],
            capture_output=True, text=True, timeout=10
        )
        if "audio" in result.stdout.lower():
            print("  AUDIO STREAM FOUND!")
        else:
            print("  No audio stream detected")
        print(result.stdout[:500] if result.stdout else "No output")
    except FileNotFoundError:
        print("  ffprobe not found - install FFmpeg")
    except Exception as e:
        print(f"  Error: {e}")

    # Method 2: Try OpenCV with audio
    print("\n[2] Testing with OpenCV...")
    try:
        import cv2
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if cap.isOpened():
            print("  Stream opened!")
            # Check properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"  Video: {width}x{height} @ {fps}fps")

            # Note: OpenCV doesn't handle audio directly
            print("  (OpenCV can't read audio - need FFmpeg/PyAudio)")
        cap.release()
    except Exception as e:
        print(f"  Error: {e}")

    # Method 3: Test audio capture with FFmpeg
    print("\n[3] Testing audio capture (5 seconds)...")
    audio_file = "tapo_test_audio.wav"
    try:
        cmd = [
            "ffmpeg", "-y",
            "-rtsp_transport", "tcp",
            "-i", url,
            "-t", "5",
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_file
        ]
        print(f"  Recording 5 seconds to {audio_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if os.path.exists(audio_file) and os.path.getsize(audio_file) > 1000:
            size = os.path.getsize(audio_file)
            print(f"  SUCCESS! Audio saved: {size} bytes")
            print(f"  Play it: start {audio_file}")
            return True
        else:
            print("  No audio captured (file too small or missing)")
            if result.stderr:
                print(f"  FFmpeg: {result.stderr[:300]}")
    except FileNotFoundError:
        print("  FFmpeg not found!")
        print("  Install: https://ffmpeg.org/download.html")
        print("  Or: choco install ffmpeg")
    except Exception as e:
        print(f"  Error: {e}")

    return False


def test_speaker(camera_ip):
    """Test if we can send audio TO camera speaker."""
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"

    print("\n[4] Testing speaker (ONVIF backchannel)...")
    print("  This is harder - Tapo may not support standard ONVIF audio back")
    print("  Usually need Tapo app or cloud API")

    # Try ONVIF
    try:
        from onvif import ONVIFCamera
        print("  Connecting via ONVIF...")
        cam = ONVIFCamera(camera_ip, 2020, 'fasspay', 'fasspay2025')

        # Check for audio output
        media = cam.create_media_service()
        profiles = media.GetProfiles()

        for p in profiles:
            print(f"  Profile: {p.Name}")
            if hasattr(p, 'AudioEncoderConfiguration'):
                print(f"    Has audio encoder!")
            if hasattr(p, 'AudioOutputConfiguration'):
                print(f"    Has audio OUTPUT! (speaker)")

    except ImportError:
        print("  ONVIF not installed: pip install onvif-zeep")
    except Exception as e:
        print(f"  ONVIF error: {e}")
        print("  (Tapo may not fully support ONVIF audio)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_tapo_audio.py <camera_ip>")
        print("Example: python test_tapo_audio.py .129")
    else:
        has_audio = test_audio_stream(sys.argv[1])
        test_speaker(sys.argv[1])

        print("\n" + "=" * 50)
        print("  SUMMARY")
        print("=" * 50)
        if has_audio:
            print("  MIC: YES - Can capture audio!")
            print("  SPEAKER: Need test with Tapo API")
            print("\n  Next: Build voice command system!")
        else:
            print("  MIC: Need FFmpeg installed")
            print("  Install FFmpeg first, then test again")
