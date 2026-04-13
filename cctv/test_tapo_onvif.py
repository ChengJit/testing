#!/usr/bin/env python3
"""
Test Tapo with Third-Party Compatibility (ONVIF) enabled!
Now speaker might be accessible!
"""

import os
import sys
import subprocess
import tempfile
import time


def test_onvif_audio(camera_ip, username, password):
    """Test ONVIF audio with third-party mode enabled."""
    print("\n[1] Testing ONVIF (Third-Party Mode)...")

    try:
        from onvif import ONVIFCamera

        # Tapo ONVIF usually on port 2020
        ports = [2020, 80, 8080, 554]

        for port in ports:
            try:
                print(f"  Trying port {port}...")
                cam = ONVIFCamera(camera_ip, port, username, password)

                # Get device info
                device = cam.create_devicemgmt_service()
                info = device.GetDeviceInformation()
                print(f"  ✓ Connected! Model: {info.Model}")

                # Get media service
                media = cam.create_media_service()
                profiles = media.GetProfiles()

                print(f"  Found {len(profiles)} profile(s)")

                for profile in profiles:
                    print(f"\n  Profile: {profile.Name} (token: {profile.token})")

                    # Check for audio
                    if hasattr(profile, 'AudioSourceConfiguration') and profile.AudioSourceConfiguration:
                        print(f"    ✓ Audio Source (mic): {profile.AudioSourceConfiguration.Name}")

                    if hasattr(profile, 'AudioEncoderConfiguration') and profile.AudioEncoderConfiguration:
                        print(f"    ✓ Audio Encoder: {profile.AudioEncoderConfiguration.Encoding}")

                    if hasattr(profile, 'AudioOutputConfiguration') and profile.AudioOutputConfiguration:
                        print(f"    ✓ Audio OUTPUT (speaker!): {profile.AudioOutputConfiguration.Name}")

                        # Try to get audio output config
                        try:
                            outputs = media.GetAudioOutputs()
                            print(f"    Audio outputs: {outputs}")
                        except:
                            pass

                    # Get stream URI
                    try:
                        stream_setup = {
                            'Stream': 'RTP-Unicast',
                            'Transport': {'Protocol': 'RTSP'}
                        }
                        uri = media.GetStreamUri({'StreamSetup': stream_setup, 'ProfileToken': profile.token})
                        print(f"    Stream: {uri.Uri}")
                    except Exception as e:
                        print(f"    Stream error: {e}")

                # Check for audio backchannel (two-way audio)
                print("\n  Checking for backchannel (two-way audio)...")
                try:
                    # Try to get audio output configurations
                    audio_outputs = media.GetAudioOutputConfigurations()
                    print(f"    Audio output configs: {audio_outputs}")

                    for ao in audio_outputs:
                        print(f"      - {ao.Name}: {ao.token}")

                except Exception as e:
                    print(f"    No audio output config: {e}")

                # Try to find backchannel
                try:
                    # Some cameras expose backchannel via GetServiceCapabilities
                    caps = media.GetServiceCapabilities()
                    print(f"    Service capabilities: {caps}")
                except:
                    pass

                return True, cam, media, profiles

            except Exception as e:
                print(f"    Port {port}: {e}")
                continue

        return False, None, None, None

    except ImportError:
        print("  Need onvif-zeep: pip install onvif-zeep")
        return False, None, None, None


def test_rtsp_backchannel(camera_ip, username, password):
    """Test RTSP backchannel for audio."""
    print("\n[2] Testing RTSP Backchannel...")

    # Create test audio
    audio_file = create_audio("Hi! How can I help?")
    if not audio_file:
        return False

    # Common backchannel URLs for Tapo with ONVIF enabled
    urls = [
        f"rtsp://{username}:{password}@{camera_ip}:554/stream1",
        f"rtsp://{username}:{password}@{camera_ip}:554/stream2",
        f"rtsp://{username}:{password}@{camera_ip}:8554/backchannel",
        f"rtsp://{username}:{password}@{camera_ip}:554/backchannel",
        f"rtsp://{username}:{password}@{camera_ip}:554/audioback",
    ]

    for url in urls:
        print(f"  Trying: {url.replace(password, '***')}")

        # First check if URL responds
        probe_cmd = ["ffprobe", "-v", "error", "-show_streams", url]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, timeout=5, text=True)
            if "audio" in result.stdout.lower():
                print(f"    ✓ Has audio stream!")
        except:
            pass

    # Try sending audio
    print("\n  Attempting to send audio...")

    # Convert to format camera might accept
    temp_audio = tempfile.mktemp(suffix=".wav")
    convert_cmd = [
        "ffmpeg", "-y", "-i", audio_file,
        "-ar", "8000", "-ac", "1",
        "-acodec", "pcm_mulaw",
        temp_audio
    ]
    subprocess.run(convert_cmd, capture_output=True)

    # Try different methods
    methods = [
        # Method 1: RTP to backchannel port
        ["ffmpeg", "-re", "-i", temp_audio, "-ar", "8000", "-ac", "1",
         "-f", "rtp", f"rtp://{camera_ip}:5000"],

        # Method 2: RTSP publish
        ["ffmpeg", "-re", "-i", temp_audio, "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw",
         "-f", "rtsp", f"rtsp://{username}:{password}@{camera_ip}:554/backchannel"],
    ]

    for cmd in methods:
        try:
            print(f"    Running: {' '.join(cmd[:5])}...")
            result = subprocess.run(cmd, capture_output=True, timeout=5)
        except:
            pass

    return False


def test_pytapo_stream(camera_ip, username, password):
    """Test pytapo media streaming."""
    print("\n[3] Testing pytapo streaming...")

    try:
        from pytapo import Tapo

        tapo = Tapo(camera_ip, username, password)

        info = tapo.getBasicInfo()
        print(f"  Connected: {info.get('device_info', {}).get('basic_info', {}).get('device_alias')}")

        # Check all methods
        methods = [m for m in dir(tapo) if not m.startswith('_')]
        audio_methods = [m for m in methods if 'audio' in m.lower() or 'speak' in m.lower() or 'talk' in m.lower() or 'stream' in m.lower()]

        if audio_methods:
            print(f"  Audio-related methods: {audio_methods}")

            for method in audio_methods:
                try:
                    func = getattr(tapo, method)
                    if callable(func):
                        result = func()
                        print(f"    {method}(): {result}")
                except Exception as e:
                    print(f"    {method}(): {e}")

        # Try to start/stop talk
        try:
            print("  Trying startTalk()...")
            tapo.startTalk() if hasattr(tapo, 'startTalk') else None
        except Exception as e:
            print(f"    {e}")

        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def create_audio(text):
    """Create audio file from text."""
    audio_file = tempfile.mktemp(suffix=".wav")

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file(text, audio_file)
        engine.runAndWait()
        engine.stop()
        return audio_file
    except:
        return None


def main():
    print("=" * 60)
    print("  TAPO THIRD-PARTY MODE TEST")
    print("  (ONVIF + Speaker Test)")
    print("=" * 60)

    # Get camera IP
    if len(sys.argv) >= 2:
        camera_ip = sys.argv[1]
    else:
        camera_ip = input("Camera IP (e.g., .129): ").strip()

    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"

    print(f"\nCamera: {camera_ip}")

    # Get ONVIF credentials (set in Tapo app under Camera Settings > Advanced > Third-Party Compatibility)
    print("\nEnter ONVIF credentials (from Tapo app > Third-Party Compatibility):")
    print("(This is usually different from your Tapo cloud login!)")
    username = input("  ONVIF Username: ").strip()
    password = input("  ONVIF Password: ").strip()

    if not username or not password:
        # Try default
        print("  Using camera credentials: fasspay / fasspay2025")
        username = "fasspay"
        password = "fasspay2025"

    # Run tests
    print("\n" + "=" * 60)

    # Test 1: ONVIF
    success, cam, media, profiles = test_onvif_audio(camera_ip, username, password)

    # Test 2: RTSP backchannel
    test_rtsp_backchannel(camera_ip, username, password)

    # Test 3: pytapo
    test_pytapo_stream(camera_ip, username, password)

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    if success:
        print("""
  ✓ ONVIF connected!

  For speaker/two-way audio, Tapo TC74 may need:
  1. Firmware update (check Tapo app)
  2. Specific ONVIF profile with audio output

  If still no speaker access, use laptop speaker!
""")
    else:
        print("""
  ONVIF connection issues.

  Check in Tapo app:
  1. Camera Settings > Advanced Settings
  2. Third-Party Compatibility > ONVIF
  3. Enable it and SET a username/password
  4. Use THAT username/password here (not cloud login!)

  Then run this test again!
""")


if __name__ == "__main__":
    main()
