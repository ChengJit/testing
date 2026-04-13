#!/usr/bin/env python3
"""Play audio to Tapo speaker via ONVIF backchannel"""

import subprocess
import requests
from requests.auth import HTTPDigestAuth
import re
import os

CAMERA_IP = "192.168.122.129"
USER = "fasspay"
PASS = "fasspay2025"

print("=" * 50)
print("  ONVIF AUDIO PLAYER")
print("=" * 50)

def get_audio_output_uri():
    """Get the ONVIF audio output/backchannel URI."""

    # Get media capabilities first
    soap = '''<?xml version="1.0" encoding="UTF-8"?>
    <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
                   xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
        <soap:Body>
            <trt:GetProfiles/>
        </soap:Body>
    </soap:Envelope>'''

    try:
        resp = requests.post(
            f"http://{CAMERA_IP}:2020/onvif/media_service",
            data=soap,
            headers={"Content-Type": "application/soap+xml"},
            auth=HTTPDigestAuth(USER, PASS),
            timeout=10
        )

        # Extract profile token
        match = re.search(r'token="([^"]+)"', resp.text)
        if match:
            profile_token = match.group(1)
            print(f"Profile token: {profile_token}")
            return profile_token
    except Exception as e:
        print(f"Error getting profiles: {e}")

    return None


def get_stream_uri(profile_token):
    """Get RTSP stream URI with backchannel."""

    soap = f'''<?xml version="1.0" encoding="UTF-8"?>
    <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
                   xmlns:trt="http://www.onvif.org/ver10/media/wsdl"
                   xmlns:tt="http://www.onvif.org/ver10/schema">
        <soap:Body>
            <trt:GetStreamUri>
                <trt:StreamSetup>
                    <tt:Stream>RTP-Unicast</tt:Stream>
                    <tt:Transport>
                        <tt:Protocol>RTSP</tt:Protocol>
                    </tt:Transport>
                </trt:StreamSetup>
                <trt:ProfileToken>{profile_token}</trt:ProfileToken>
            </trt:GetStreamUri>
        </soap:Body>
    </soap:Envelope>'''

    try:
        resp = requests.post(
            f"http://{CAMERA_IP}:2020/onvif/media_service",
            data=soap,
            headers={"Content-Type": "application/soap+xml"},
            auth=HTTPDigestAuth(USER, PASS),
            timeout=10
        )

        match = re.search(r'<tt:Uri>([^<]+)</tt:Uri>', resp.text)
        if match:
            uri = match.group(1)
            print(f"Stream URI: {uri}")
            return uri
    except Exception as e:
        print(f"Error: {e}")

    return None


def get_audio_outputs():
    """Get audio output configurations."""

    soap = '''<?xml version="1.0" encoding="UTF-8"?>
    <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
                   xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
        <soap:Body>
            <trt:GetAudioOutputs/>
        </soap:Body>
    </soap:Envelope>'''

    try:
        resp = requests.post(
            f"http://{CAMERA_IP}:2020/onvif/media_service",
            data=soap,
            headers={"Content-Type": "application/soap+xml"},
            auth=HTTPDigestAuth(USER, PASS),
            timeout=10
        )
        print(f"Audio outputs response: {resp.status_code}")

        # Check for audio output token
        match = re.search(r'token="([^"]+)"', resp.text)
        if match:
            print(f"Audio output token: {match.group(1)}")
            return match.group(1)

        if "AudioOutput" in resp.text:
            print("Audio output found in response")
            print(resp.text[:1000])

    except Exception as e:
        print(f"Error: {e}")

    return None


def play_audio_via_ffmpeg(audio_file):
    """Send audio to camera using FFmpeg RTSP ANNOUNCE."""

    print(f"\nPlaying: {audio_file}")

    # RTSP URL with credentials
    rtsp_url = f"rtsp://{USER}:{PASS}@{CAMERA_IP}:554/stream1"

    # Method 1: Try sending as RTSP output
    print("\n[Method 1] FFmpeg RTSP push...")
    cmd = [
        "ffmpeg", "-y",
        "-re",
        "-i", audio_file,
        "-ar", "8000",
        "-ac", "1",
        "-c:a", "pcm_mulaw",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        f"rtsp://{USER}:{PASS}@{CAMERA_IP}:554/audioback"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print(f"Result: {result.returncode}")
        if result.stderr:
            print(f"FFmpeg: {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        print("Timeout (may be ok)")
    except Exception as e:
        print(f"Error: {e}")

    # Method 2: Try backchannel port
    print("\n[Method 2] Backchannel direct...")
    cmd = [
        "ffmpeg", "-y",
        "-re",
        "-i", audio_file,
        "-ar", "8000",
        "-ac", "1",
        "-c:a", "pcm_alaw",
        "-f", "rtp",
        f"rtp://{CAMERA_IP}:5000"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print(f"Result: {result.returncode}")
    except:
        pass


def create_test_audio():
    """Create a test beep sound."""
    test_file = "test_beep.wav"

    if not os.path.exists(test_file):
        print("Creating test beep...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "sine=frequency=800:duration=3",
            "-ar", "8000",
            "-ac", "1",
            test_file
        ]
        subprocess.run(cmd, capture_output=True)

    return test_file


def main():
    # Get ONVIF info
    profile = get_audio_output_uri()
    if profile:
        get_stream_uri(profile)

    get_audio_outputs()

    # Create test audio
    audio_file = create_test_audio()

    # Try to play
    play_audio_via_ffmpeg(audio_file)

    print("\n" + "=" * 50)
    print("Did you hear anything from the camera speaker?")
    print("=" * 50)


if __name__ == "__main__":
    main()
