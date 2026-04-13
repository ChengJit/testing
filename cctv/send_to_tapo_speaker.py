#!/usr/bin/env python3
"""
Send Audio to Tapo Speaker - Multiple Methods
"""

import os
import sys
import subprocess
import tempfile
import time
import socket


def create_audio(text="Hi! How can I help?"):
    """Create audio file."""
    audio_file = os.path.join(os.path.dirname(__file__), "response.wav")

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file(text, audio_file)
        engine.runAndWait()
        engine.stop()
        time.sleep(0.3)
        print(f"Created: {audio_file}")
        return audio_file
    except Exception as e:
        print(f"Audio creation error: {e}")
        return None


def convert_audio(input_file, format_type="mulaw"):
    """Convert audio to camera-compatible format."""
    output_file = tempfile.mktemp(suffix=".wav")

    if format_type == "mulaw":
        cmd = [
            "ffmpeg", "-y", "-i", input_file,
            "-ar", "8000", "-ac", "1", "-acodec", "pcm_mulaw",
            output_file
        ]
    elif format_type == "alaw":
        cmd = [
            "ffmpeg", "-y", "-i", input_file,
            "-ar", "8000", "-ac", "1", "-acodec", "pcm_alaw",
            output_file
        ]
    else:  # pcm
        cmd = [
            "ffmpeg", "-y", "-i", input_file,
            "-ar", "8000", "-ac", "1", "-acodec", "pcm_s16le",
            output_file
        ]

    subprocess.run(cmd, capture_output=True)
    return output_file


def method1_rtsp_backchannel(camera_ip, username, password, audio_file):
    """Try RTSP backchannel."""
    print("\n[Method 1] RTSP Backchannel...")

    # Convert audio
    audio_mulaw = convert_audio(audio_file, "mulaw")

    # Try various backchannel endpoints
    endpoints = [
        "/backchannel",
        "/audioback",
        "/audio/backchannel",
        "/talk",
        "/stream1/audio",
    ]

    for endpoint in endpoints:
        url = f"rtsp://{username}:{password}@{camera_ip}:554{endpoint}"
        print(f"  Trying: {endpoint}")

        cmd = [
            "ffmpeg", "-re",
            "-i", audio_mulaw,
            "-ar", "8000", "-ac", "1",
            "-c:a", "pcm_mulaw",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5, text=True)
            if "200 OK" in result.stderr or result.returncode == 0:
                print(f"    ✓ Might have worked!")
                return True
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            pass

    return False


def method2_rtp_stream(camera_ip, audio_file):
    """Try RTP stream to camera."""
    print("\n[Method 2] RTP Audio Stream...")

    audio_mulaw = convert_audio(audio_file, "mulaw")

    # Common audio RTP ports
    ports = [5000, 5002, 5004, 6000, 6002, 8000, 8002]

    for port in ports:
        print(f"  Trying RTP to port {port}...")

        cmd = [
            "ffmpeg", "-re",
            "-i", audio_mulaw,
            "-ar", "8000", "-ac", "1",
            "-c:a", "pcm_mulaw",
            "-f", "rtp",
            f"rtp://{camera_ip}:{port}"
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=3)
        except:
            pass

    return False


def method3_onvif_backchannel(camera_ip, username, password, audio_file):
    """Try ONVIF backchannel."""
    print("\n[Method 3] ONVIF Backchannel...")

    try:
        from onvif import ONVIFCamera

        cam = ONVIFCamera(camera_ip, 2020, username, password)
        media = cam.create_media_service()

        # Get profiles
        profiles = media.GetProfiles()
        print(f"  Found {len(profiles)} profiles")

        for profile in profiles:
            # Check for audio output
            if hasattr(profile, 'AudioOutputConfiguration') and profile.AudioOutputConfiguration:
                print(f"  ✓ Profile {profile.Name} has audio output!")

                # Try to get backchannel URI
                try:
                    # Some cameras support GetAudioOutputConfigurationOptions
                    opts = media.GetAudioOutputConfigurationOptions({'ProfileToken': profile.token})
                    print(f"    Audio output options: {opts}")
                except:
                    pass

        # Try Media2 service (newer ONVIF)
        try:
            media2 = cam.create_media2_service()
            profiles2 = media2.GetProfiles()
            print(f"  Media2 profiles: {len(profiles2)}")
        except:
            print("  No Media2 service")

        return False

    except Exception as e:
        print(f"  Error: {e}")
        return False


def method4_http_api(camera_ip, username, password, audio_file):
    """Try HTTP API for audio."""
    print("\n[Method 4] HTTP Audio API...")

    import requests
    from requests.auth import HTTPDigestAuth, HTTPBasicAuth

    # Common API endpoints for two-way audio
    endpoints = [
        "/cgi-bin/audio.cgi",
        "/api/audio/talk",
        "/audio/talk",
        "/ISAPI/System/Audio/channels/1/talkback",
        "/cgi-bin/intercom.cgi",
    ]

    for endpoint in endpoints:
        url = f"http://{camera_ip}{endpoint}"
        print(f"  Trying: {endpoint}")

        for auth in [HTTPDigestAuth(username, password), HTTPBasicAuth(username, password)]:
            try:
                # Try GET first to see if endpoint exists
                resp = requests.get(url, auth=auth, timeout=2)
                if resp.status_code != 404:
                    print(f"    Response: {resp.status_code}")

                    # Try POST with audio
                    with open(audio_file, 'rb') as f:
                        audio_data = f.read()
                    resp = requests.post(url, auth=auth, data=audio_data, timeout=5,
                                        headers={'Content-Type': 'audio/wav'})
                    print(f"    POST: {resp.status_code}")

            except:
                pass

    return False


def method5_pytapo_talkback(camera_ip, username, password, audio_file):
    """Try pytapo talkback."""
    print("\n[Method 5] Pytapo Talkback...")

    try:
        from pytapo import Tapo

        tapo = Tapo(camera_ip, username, password)
        print("  Connected!")

        # List all methods related to audio/talk
        methods = dir(tapo)
        talk_methods = [m for m in methods if 'talk' in m.lower() or 'audio' in m.lower() or 'speak' in m.lower()]
        print(f"  Talk-related methods: {talk_methods}")

        # Try each method
        for method_name in talk_methods:
            try:
                method = getattr(tapo, method_name)
                if callable(method):
                    print(f"  Calling {method_name}()...")
                    result = method()
                    print(f"    Result: {result}")
            except Exception as e:
                print(f"    {method_name}: {e}")

        return False

    except Exception as e:
        print(f"  Error: {e}")
        return False


def method6_play_laptop_speaker(audio_file):
    """Fallback: Play on laptop speaker."""
    print("\n[Method 6] Laptop Speaker (Fallback)...")

    try:
        if sys.platform == "win32":
            # Use Windows Media Player silently
            subprocess.Popen(
                ['powershell', '-c', f'(New-Object Media.SoundPlayer "{audio_file}").PlaySync()'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            os.system(f'aplay "{audio_file}" 2>/dev/null || afplay "{audio_file}" 2>/dev/null &')

        print("  ✓ Playing on laptop speaker!")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    print("=" * 60)
    print("  SEND AUDIO TO TAPO SPEAKER")
    print("=" * 60)

    # Get camera info
    if len(sys.argv) >= 2:
        camera_ip = sys.argv[1]
    else:
        camera_ip = input("Camera IP (.129): ").strip()

    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"

    username = "fasspay"
    password = "fasspay2025"

    print(f"\nCamera: {camera_ip}")
    print(f"Credentials: {username}")

    # Create audio
    print("\n" + "-" * 40)
    audio_file = create_audio("Hi! How can I help?")
    if not audio_file:
        print("Failed to create audio!")
        return

    # Try all methods
    print("\n" + "=" * 60)
    print("  TESTING METHODS")
    print("=" * 60)

    success = False

    success = method1_rtsp_backchannel(camera_ip, username, password, audio_file) or success
    success = method2_rtp_stream(camera_ip, audio_file) or success
    success = method3_onvif_backchannel(camera_ip, username, password, audio_file) or success
    success = method4_http_api(camera_ip, username, password, audio_file) or success
    success = method5_pytapo_talkback(camera_ip, username, password, audio_file) or success

    # Summary
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)

    if success:
        print("  ✓ Found a way to send audio to camera speaker!")
    else:
        print("  ✗ Camera speaker not directly accessible")
        print("\n  Using laptop speaker instead...")
        method6_play_laptop_speaker(audio_file)

        print("""
  RECOMMENDATION:
  ---------------
  Tapo TC74 speaker requires proprietary protocol.

  Best solution: Put a small speaker near the rack
  connected to your laptop. Voice feedback will
  come from there!

  Or use a Bluetooth speaker near the camera.
  """)


if __name__ == "__main__":
    main()
