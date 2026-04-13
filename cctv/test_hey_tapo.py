#!/usr/bin/env python3
"""
Simple "Hey Tapo" Test
- Listen from Tapo mic
- Reply through Tapo speaker: "Hi, how can I help?"
"""

import os
import sys
import time
import subprocess
import tempfile
import threading

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


def generate_response_audio():
    """Generate 'Hi how can I help' audio file."""
    audio_file = os.path.join(os.path.dirname(__file__), "hey_tapo_response.wav")

    if os.path.exists(audio_file):
        return audio_file

    print("Generating response audio...")

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file("Hi! How can I help?", audio_file)
        engine.runAndWait()
        print(f"Saved: {audio_file}")
        return audio_file
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None


def listen_for_wake_word(camera_ip, duration=4):
    """Listen to camera mic and check for 'hey tapo'."""
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream1"
    temp_file = tempfile.mktemp(suffix=".wav")

    # Record audio
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", url,
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        temp_file
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=duration + 5)

        if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 1000:
            return None

        # Recognize speech
        import speech_recognition as sr
        recognizer = sr.Recognizer()

        with sr.AudioFile(temp_file) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio).lower()
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"Speech API error: {e}")
            return ""

    except Exception as e:
        print(f"Listen error: {e}")
        return None
    finally:
        try:
            os.remove(temp_file)
        except:
            pass


def send_audio_to_tapo(camera_ip, audio_file):
    """Send audio to Tapo camera speaker."""
    print("Sending audio to Tapo speaker...")

    # Method 1: Try pytapo
    try:
        from pytapo import Tapo

        print("  Trying pytapo...")
        tapo = Tapo(camera_ip, "fasspay", "fasspay2025")

        # Check if camera supports two-way audio
        info = tapo.getBasicInfo()
        print(f"  Camera: {info.get('device_info', {}).get('basic_info', {}).get('device_model', 'Unknown')}")

        # Try to send audio (this might not work on all models)
        # pytapo uses ONVIF backchannel
        print("  Attempting audio stream...")

        # For now, pytapo might not have direct speaker method
        # Let's try ONVIF approach
        raise Exception("Try ONVIF")

    except Exception as e:
        print(f"  pytapo: {e}")

    # Method 2: Try ONVIF backchannel
    try:
        print("  Trying ONVIF backchannel...")

        # Convert audio to format Tapo expects (G.711 u-law or a-law)
        temp_audio = tempfile.mktemp(suffix=".wav")
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_file,
            "-ar", "8000",
            "-ac", "1",
            "-acodec", "pcm_mulaw",
            temp_audio
        ]
        subprocess.run(cmd, capture_output=True)

        # Try sending via RTSP (backchannel)
        # This is tricky - need proper ONVIF setup
        # Most Tapo cameras use proprietary protocol

        print("  ONVIF backchannel requires more setup...")
        raise Exception("Try FFmpeg direct")

    except Exception as e:
        print(f"  ONVIF: {e}")

    # Method 3: Try FFmpeg direct stream
    try:
        print("  Trying FFmpeg stream to camera...")

        # Some cameras accept audio on specific endpoint
        rtsp_out = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream1"

        # This usually doesn't work for Tapo but worth trying
        cmd = [
            "ffmpeg", "-y",
            "-re",
            "-i", audio_file,
            "-acodec", "aac",
            "-f", "rtsp",
            rtsp_out
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)

        if result.returncode == 0:
            print("  Audio sent!")
            return True

    except Exception as e:
        print(f"  FFmpeg: {e}")

    # Method 4: Fallback to laptop speaker
    print("\n  Tapo speaker not accessible directly.")
    print("  Playing through laptop speaker instead...")

    try:
        if sys.platform == "win32":
            os.system(f'start /min wmplayer "{audio_file}"')
        else:
            os.system(f'aplay "{audio_file}" 2>/dev/null || afplay "{audio_file}" 2>/dev/null')
        return True
    except:
        pass

    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_hey_tapo.py <camera_ip>")
        print("Example: python test_hey_tapo.py .129")
        return

    camera_ip = sys.argv[1]
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"

    print("=" * 50)
    print("  HEY TAPO TEST")
    print("=" * 50)
    print(f"\nCamera: {camera_ip}")

    # Generate response audio
    response_audio = generate_response_audio()
    if not response_audio:
        print("Failed to generate response audio!")
        return

    # Install speech recognition if needed
    try:
        import speech_recognition
    except ImportError:
        print("Installing speech_recognition...")
        os.system("pip install SpeechRecognition")
        import speech_recognition

    print("\n" + "=" * 50)
    print("  LISTENING... Say 'Hey Tapo'!")
    print("=" * 50)

    while True:
        print("\n[Listening...]", end=" ", flush=True)

        text = listen_for_wake_word(camera_ip, duration=4)

        if text is None:
            print("Error - retrying...")
            time.sleep(1)
            continue

        if text:
            print(f"Heard: '{text}'")

            if "hey tapo" in text or "a tapo" in text or "hey taco" in text:
                print("\n*** WAKE WORD DETECTED! ***")

                # Send response
                send_audio_to_tapo(camera_ip, response_audio)

                print("\n[Listening again...]")
        else:
            print("(silence)")

        # Small delay
        time.sleep(0.5)


if __name__ == "__main__":
    main()
