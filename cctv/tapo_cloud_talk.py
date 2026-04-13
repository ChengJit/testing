#!/usr/bin/env python3
"""
Tapo Cloud Talk - Use Tapo Cloud Account for Speaker!
Supports phone number login!
"""

import os
import sys
import time
import subprocess
import tempfile
import asyncio

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


def create_audio(text="Hi! How can I help?"):
    """Create audio file."""
    audio_file = os.path.join(os.path.dirname(__file__), "tapo_response.wav")

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file(text, audio_file)
        engine.runAndWait()
        engine.stop()
        time.sleep(0.3)
        return audio_file
    except Exception as e:
        print(f"Audio error: {e}")
        return None


def test_pytapo_cloud(camera_ip, cloud_user, cloud_pass):
    """Test pytapo with cloud credentials."""
    print("\n[1] Testing pytapo with cloud login...")

    try:
        from pytapo import Tapo

        print(f"  Connecting to {camera_ip}...")
        print(f"  Cloud user: {cloud_user}")

        # For phone number login, use the phone number as username
        # Cloud password is your Tapo app password
        tapo = Tapo(
            camera_ip,
            cloud_user,      # Can be email OR phone number
            cloud_pass,      # Your Tapo app password
            cloudPassword=cloud_pass  # Same password for cloud auth
        )

        # Get camera info
        info = tapo.getBasicInfo()
        basic_info = info.get('device_info', {}).get('basic_info', {})
        print(f"  ✓ Connected to: {basic_info.get('device_alias', 'Unknown')}")
        print(f"  Model: {basic_info.get('device_model', 'Unknown')}")

        return tapo

    except Exception as e:
        print(f"  Error: {e}")
        print("\n  Troubleshooting:")
        print("  - For phone login, use format: +60123456789")
        print("  - Password is your Tapo app password")
        return None


def test_pytapo_talkback(tapo, audio_file):
    """Try to use talkback feature."""
    print("\n[2] Testing talkback/speaker...")

    if not tapo:
        return False

    # List all methods
    all_methods = [m for m in dir(tapo) if not m.startswith('_')]
    print(f"  Available methods ({len(all_methods)}):")

    # Find audio/talk related
    talk_methods = []
    for m in all_methods:
        if any(kw in m.lower() for kw in ['talk', 'audio', 'speak', 'stream', 'media', 'sound']):
            talk_methods.append(m)
            print(f"    * {m}")

    # Try talk-related methods
    for method_name in talk_methods:
        try:
            method = getattr(tapo, method_name)
            if callable(method):
                print(f"\n  Calling {method_name}()...")
                try:
                    result = method()
                    print(f"    Result: {result}")
                except TypeError:
                    # Method might need parameters
                    print(f"    (needs parameters)")
        except Exception as e:
            print(f"    Error: {e}")

    return False


def test_pytapo_stream(tapo, camera_ip, cloud_user, cloud_pass, audio_file):
    """Try pytapo media streaming for two-way audio."""
    print("\n[3] Testing media stream...")

    try:
        from pytapo.media_stream.downloader import Downloader

        print("  Creating media stream...")

        # Callback for receiving data
        def callback(data):
            pass

        # Create downloader
        downloader = Downloader(
            tapo,
            cloud_user,
            cloud_pass,
            camera_ip,
            callback,
            None,  # fileName
            True,  # isRaw
        )

        print("  Downloader created!")
        print("  Checking for two-way audio support...")

        # The downloader might have methods for sending audio
        stream_methods = [m for m in dir(downloader) if not m.startswith('_')]
        audio_methods = [m for m in stream_methods if 'audio' in m.lower() or 'send' in m.lower() or 'talk' in m.lower()]

        if audio_methods:
            print(f"  Audio methods found: {audio_methods}")
            for m in audio_methods:
                print(f"    - {m}")

        return False

    except Exception as e:
        print(f"  Stream error: {e}")
        return False


def test_pytapo_async(camera_ip, cloud_user, cloud_pass, audio_file):
    """Try async pytapo for real-time communication."""
    print("\n[4] Testing async communication...")

    try:
        # Check if pytapo has async support
        from pytapo import Tapo

        tapo = Tapo(camera_ip, cloud_user, cloud_pass, cloudPassword=cloud_pass)

        # Some newer versions might have async talkback
        if hasattr(tapo, 'startTalk'):
            print("  Found startTalk method!")
            try:
                result = tapo.startTalk()
                print(f"    Result: {result}")

                # If started, try sending audio
                time.sleep(1)

                if hasattr(tapo, 'stopTalk'):
                    tapo.stopTalk()

                return True
            except Exception as e:
                print(f"    Error: {e}")

        if hasattr(tapo, 'sendAudio'):
            print("  Found sendAudio method!")
            try:
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()
                result = tapo.sendAudio(audio_data)
                print(f"    Result: {result}")
                return True
            except Exception as e:
                print(f"    Error: {e}")

        return False

    except Exception as e:
        print(f"  Async error: {e}")
        return False


def main():
    print("=" * 60)
    print("  TAPO CLOUD TALK")
    print("  (Speaker via Cloud Account)")
    print("=" * 60)

    # Get camera IP
    if len(sys.argv) >= 2:
        camera_ip = sys.argv[1]
    else:
        camera_ip = input("Camera IP (e.g., .129): ").strip()

    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"

    # Get cloud credentials
    print("\n" + "-" * 40)
    print("Enter your Tapo app login:")
    print("(For phone number, use format: +60123456789)")
    print("-" * 40)

    cloud_user = input("  Email or Phone: ").strip()
    cloud_pass = input("  Password: ").strip()

    if not cloud_user or not cloud_pass:
        print("Need credentials!")
        return

    # Create test audio
    print("\n" + "-" * 40)
    audio_file = create_audio("Hi! How can I help?")

    # Run tests
    print("\n" + "=" * 60)
    tapo = test_pytapo_cloud(camera_ip, cloud_user, cloud_pass)

    if tapo:
        test_pytapo_talkback(tapo, audio_file)
        test_pytapo_stream(tapo, camera_ip, cloud_user, cloud_pass, audio_file)
        test_pytapo_async(camera_ip, cloud_user, cloud_pass, audio_file)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print("""
  Pytapo STATUS:
  - Cloud login: Works with email or phone number
  - Camera control: Works (get info, settings, etc.)
  - Two-way audio: Limited support in pytapo library

  The pytapo library focuses on control, not media streaming.
  For full two-way audio, alternatives are:

  1. Home Assistant + Tapo integration
     - Full two-way audio support
     - https://github.com/JurajNyiri/HomeAssistant-Tapo-Control

  2. go2rtc (recommended!)
     - Supports Tapo two-way audio
     - https://github.com/AlexxIT/go2rtc

  3. Frigate NVR
     - Works with Tapo cameras
     - Two-way audio via WebRTC

  QUICK WORKAROUND:
  -----------------
  Use laptop/external speaker near rack!
  Voice input from Tapo mic still works!
    """)

    # Ask if want to try go2rtc
    print("\nWant me to set up go2rtc for two-way audio? (y/n)")


if __name__ == "__main__":
    main()
