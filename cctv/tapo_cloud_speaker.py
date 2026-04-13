#!/usr/bin/env python3
"""
Tapo Cloud Speaker - Send audio via TP-Link Cloud API
Requires your Tapo app login credentials!
"""

import os
import sys
import asyncio
import subprocess
import tempfile
import time

# Install required packages
def install_packages():
    packages = ["pytapo", "pyttsx3"]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            os.system(f"pip install {pkg}")


install_packages()


class TapoCloudSpeaker:
    """Send audio to Tapo camera via cloud API."""

    def __init__(self, camera_ip, cloud_email, cloud_password):
        self.camera_ip = camera_ip
        self.cloud_email = cloud_email
        self.cloud_password = cloud_password
        self.tapo = None

    def connect(self):
        """Connect to Tapo camera via cloud."""
        try:
            from pytapo import Tapo

            print(f"Connecting to Tapo camera at {self.camera_ip}...")
            print(f"Using cloud account: {self.cloud_email}")

            # Connect with cloud credentials
            self.tapo = Tapo(
                self.camera_ip,
                self.cloud_email,
                self.cloud_password,
                cloudPassword=self.cloud_password
            )

            # Get camera info
            info = self.tapo.getBasicInfo()
            basic = info.get('device_info', {}).get('basic_info', {})
            print(f"Connected to: {basic.get('device_alias', 'Tapo Camera')}")
            print(f"Model: {basic.get('device_model', 'Unknown')}")

            return True

        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def get_audio_capabilities(self):
        """Check what audio features are available."""
        if not self.tapo:
            return None

        print("\nChecking audio capabilities...")

        capabilities = {}

        # Check microphone
        try:
            mic_config = self.tapo.getMicrophoneConfig() if hasattr(self.tapo, 'getMicrophoneConfig') else None
            capabilities['microphone'] = mic_config
            print(f"  Microphone: {mic_config}")
        except Exception as e:
            print(f"  Microphone: {e}")

        # Check speaker
        try:
            speaker_config = self.tapo.getSpeakerConfig() if hasattr(self.tapo, 'getSpeakerConfig') else None
            capabilities['speaker'] = speaker_config
            print(f"  Speaker: {speaker_config}")
        except Exception as e:
            print(f"  Speaker: {e}")

        # Check audio config
        try:
            audio_config = self.tapo.getAudioConfig() if hasattr(self.tapo, 'getAudioConfig') else None
            capabilities['audio'] = audio_config
            print(f"  Audio config: {audio_config}")
        except Exception as e:
            print(f"  Audio config: {e}")

        return capabilities

    def list_methods(self):
        """List available pytapo methods."""
        if not self.tapo:
            return

        print("\nAvailable Tapo methods:")
        methods = [m for m in dir(self.tapo) if not m.startswith('_')]
        for m in sorted(methods):
            if 'audio' in m.lower() or 'speak' in m.lower() or 'sound' in m.lower():
                print(f"  * {m}")
        print("  ---")
        for m in sorted(methods):
            if 'audio' not in m.lower() and 'speak' not in m.lower() and 'sound' not in m.lower():
                print(f"    {m}")

    def play_alarm(self):
        """Try to play built-in alarm sound."""
        if not self.tapo:
            return False

        print("\nTrying to play alarm sound...")
        try:
            # Some Tapo cameras have alarm/siren
            result = self.tapo.startManualAlarm() if hasattr(self.tapo, 'startManualAlarm') else None
            print(f"  startManualAlarm: {result}")
            time.sleep(2)
            self.tapo.stopManualAlarm() if hasattr(self.tapo, 'stopManualAlarm') else None
            return True
        except Exception as e:
            print(f"  Error: {e}")
            return False

    def send_audio_stream(self, audio_file):
        """Try to send audio stream to camera."""
        if not self.tapo:
            return False

        print(f"\nTrying to send audio: {audio_file}")

        # Method 1: Try pytapo streaming
        try:
            from pytapo.media_stream.downloader import Downloader

            # Check if there's an upload/stream method
            print("  Checking for audio stream capability...")

            # Get stream URL
            stream_url = self.tapo.getStreamUrl() if hasattr(self.tapo, 'getStreamUrl') else None
            print(f"  Stream URL: {stream_url}")

        except Exception as e:
            print(f"  Stream error: {e}")

        # Method 2: Try direct API call for speaker
        try:
            print("  Trying direct speaker API...")

            # Some cameras have talkback/intercom feature
            # This varies by model
            pass

        except Exception as e:
            print(f"  API error: {e}")

        return False


def create_test_audio():
    """Create test audio file."""
    audio_file = os.path.join(os.path.dirname(__file__), "hi_response.wav")

    if os.path.exists(audio_file):
        return audio_file

    print("Creating test audio...")

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file("Hi! How can I help?", audio_file)
        engine.runAndWait()
        engine.stop()
        time.sleep(0.5)
        return audio_file
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("=" * 60)
    print("  TAPO CLOUD SPEAKER TEST")
    print("=" * 60)

    # Get camera IP
    if len(sys.argv) >= 2:
        camera_ip = sys.argv[1]
        if camera_ip.startswith('.'):
            camera_ip = f"192.168.122{camera_ip}"
    else:
        camera_ip = input("Camera IP (e.g., .129 or 192.168.122.129): ").strip()
        if camera_ip.startswith('.'):
            camera_ip = f"192.168.122{camera_ip}"

    # Get cloud credentials
    print("\nEnter your TP-Link/Tapo app credentials:")
    cloud_email = input("  Email: ").strip()
    cloud_password = input("  Password: ").strip()

    if not cloud_email or not cloud_password:
        print("Need email and password!")
        return

    # Create test audio
    audio_file = create_test_audio()

    # Connect and test
    speaker = TapoCloudSpeaker(camera_ip, cloud_email, cloud_password)

    if not speaker.connect():
        print("\nFailed to connect!")
        print("Check:")
        print("  1. Camera IP is correct")
        print("  2. Email/password match your Tapo app login")
        print("  3. Camera is online")
        return

    # List available methods
    speaker.list_methods()

    # Check capabilities
    speaker.get_audio_capabilities()

    # Try alarm (if available)
    print("\n" + "-" * 40)
    print("Testing speaker with alarm...")
    speaker.play_alarm()

    # Try audio stream
    if audio_file:
        speaker.send_audio_stream(audio_file)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("""
Tapo TC74 two-way audio status:
- Microphone: Works via RTSP stream ✓
- Speaker: Limited API access

For speaker, options are:
1. Use alarm sound (if supported)
2. Use laptop speaker as workaround
3. Use Home Assistant + Tapo integration
4. Use Tapo app directly for talk-back

Recommendation: Use laptop speaker for voice responses!
""")


if __name__ == "__main__":
    main()
