#!/usr/bin/env python3
"""
Test Tapo TC74 Speaker - Send audio TO camera
"""

import os
import sys
import subprocess
import tempfile
import time


def test_pytapo_speaker(camera_ip, audio_file):
    """Try pytapo for two-way audio."""
    print("\n[1] Testing pytapo...")

    try:
        from pytapo import Tapo
        from pytapo.media_stream.downloader import Downloader

        tapo = Tapo(camera_ip, "fasspay", "fasspay2025")

        # Get camera info
        info = tapo.getBasicInfo()
        basic = info.get('device_info', {}).get('basic_info', {})
        print(f"  Model: {basic.get('device_model', 'Unknown')}")
        print(f"  Name: {basic.get('device_alias', 'Unknown')}")

        # Check audio capabilities
        print("  Checking audio capabilities...")

        # Try to get audio config
        try:
            audio_config = tapo.getAudioConfig()
            print(f"  Audio config: {audio_config}")
        except:
            print("  No audio config method")

        # Try media stream with audio
        print("  pytapo may support two-way audio via stream...")
        print("  (This requires more complex setup)")

        return False

    except ImportError:
        print("  pytapo not installed")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_onvif_speaker(camera_ip, audio_file):
    """Try ONVIF backchannel audio."""
    print("\n[2] Testing ONVIF backchannel...")

    try:
        from onvif import ONVIFCamera

        # Try different ports (2020 is common for Tapo)
        for port in [2020, 80, 8080]:
            try:
                print(f"  Trying port {port}...")
                cam = ONVIFCamera(camera_ip, port, 'fasspay', 'fasspay2025')

                # Get device info
                device = cam.create_devicemgmt_service()
                info = device.GetDeviceInformation()
                print(f"  Found: {info.Model}")

                # Check for audio outputs
                media = cam.create_media_service()
                profiles = media.GetProfiles()

                for p in profiles:
                    print(f"  Profile: {p.Name}")
                    if hasattr(p, 'AudioOutputConfiguration') and p.AudioOutputConfiguration:
                        print(f"    HAS AUDIO OUTPUT!")
                        # Try to get audio output URI
                        try:
                            stream_uri = media.GetStreamUri({
                                'StreamSetup': {
                                    'Stream': 'RTP-Unicast',
                                    'Transport': {'Protocol': 'RTSP'}
                                },
                                'ProfileToken': p.token
                            })
                            print(f"    Stream URI: {stream_uri.Uri}")
                        except:
                            pass

                return False

            except Exception as e:
                continue

        print("  ONVIF not responding on known ports")
        return False

    except ImportError:
        print("  onvif-zeep not installed")
        print("  Install: pip install onvif-zeep")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_tapo_cloud_api(camera_ip):
    """Check Tapo cloud API (requires cloud account)."""
    print("\n[3] Tapo Cloud API...")
    print("  Two-way audio usually requires Tapo app or cloud API")
    print("  This needs your TP-Link cloud credentials")
    print("  (Not implemented - use Tapo app for speaker)")
    return False


def test_ffmpeg_backchannel(camera_ip, audio_file):
    """Try FFmpeg RTSP backchannel."""
    print("\n[4] Testing FFmpeg backchannel...")

    # Convert audio to proper format
    temp_audio = tempfile.mktemp(suffix=".wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_file,
        "-ar", "8000",
        "-ac", "1",
        "-acodec", "pcm_mulaw",
        "-f", "wav",
        temp_audio
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except Exception as e:
        print(f"  Audio conversion failed: {e}")
        return False

    # Try sending to backchannel
    # Tapo cameras might use /audioback or /backchannel
    urls_to_try = [
        f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream1/backchannel",
        f"rtsp://fasspay:fasspay2025@{camera_ip}:554/audioback",
        f"rtsp://fasspay:fasspay2025@{camera_ip}:554/backchannel",
    ]

    for url in urls_to_try:
        print(f"  Trying: {url}")
        cmd = [
            "ffmpeg", "-y",
            "-re",
            "-i", temp_audio,
            "-c:a", "pcm_mulaw",
            "-ar", "8000",
            "-ac", "1",
            "-f", "rtp",
            url
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode == 0:
                print("  Success!")
                return True
        except:
            pass

    print("  No backchannel endpoint found")

    # Cleanup
    try:
        os.remove(temp_audio)
    except:
        pass

    return False


def create_test_audio():
    """Create test audio file."""
    audio_file = os.path.join(os.path.dirname(__file__), "test_response.wav")

    if os.path.exists(audio_file):
        return audio_file

    print("Creating test audio 'Hi, how can I help?'...")

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file("Hi! How can I help?", audio_file)
        engine.runAndWait()
        engine.stop()
        print(f"  Created: {audio_file}")
        return audio_file
    except Exception as e:
        print(f"  Error: {e}")

        # Try with espeak/say fallback
        try:
            cmd = ["espeak", "-w", audio_file, "Hi, how can I help?"]
            subprocess.run(cmd, check=True)
            return audio_file
        except:
            pass

    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_tapo_speaker.py <camera_ip>")
        print("Example: python test_tapo_speaker.py .129")
        return

    camera_ip = sys.argv[1]
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"

    print("=" * 50)
    print("  TAPO SPEAKER TEST")
    print("=" * 50)
    print(f"Camera: {camera_ip}")

    # Create test audio
    audio_file = create_test_audio()
    if not audio_file:
        print("\nFailed to create test audio!")
        return

    # Try different methods
    success = False

    success = test_pytapo_speaker(camera_ip, audio_file) or success
    success = test_onvif_speaker(camera_ip, audio_file) or success
    success = test_ffmpeg_backchannel(camera_ip, audio_file) or success
    test_tapo_cloud_api(camera_ip)

    print("\n" + "=" * 50)
    print("  RESULT")
    print("=" * 50)

    if success:
        print("  Speaker access: SUCCESS!")
    else:
        print("  Speaker access: NOT AVAILABLE directly")
        print("\n  Tapo TC74 two-way audio typically requires:")
        print("  1. Tapo app (official)")
        print("  2. TP-Link cloud API (complex)")
        print("  3. Home Assistant Tapo integration")
        print("\n  WORKAROUND: Use laptop speaker for voice feedback")
        print("  (The mic still works for voice input!)")


if __name__ == "__main__":
    main()
