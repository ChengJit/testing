#!/usr/bin/env python3
"""
TAPO SPEAKER PLUGIN
Send audio files directly to Tapo camera speaker via WebRTC!

Uses go2rtc as relay with WebRTC backchannel.
"""

import asyncio
import json
import os
import sys
import time
import fractions

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import av


GO2RTC_URL = "http://localhost:1984"
CAMERA_STREAM = "tapo_hd"  # Stream name in go2rtc


class AudioFileTrack(MediaStreamTrack):
    """Stream audio from file to WebRTC."""

    kind = "audio"
    SAMPLES_PER_FRAME = 960  # 20ms at 48kHz

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.container = None
        self.stream = None
        self.resampler = None
        self._start_time = None
        self._frame_count = 0
        self._buffer = b''  # Buffer for audio samples
        self._packets = None

    async def recv(self):
        """Get next audio frame."""
        import numpy as np

        if self.container is None:
            # Open audio file
            self.container = av.open(self.file_path)
            self.stream = self.container.streams.audio[0]

            # Resample to 48kHz mono (WebRTC standard)
            self.resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=48000
            )
            self._start_time = time.time()
            self._packets = self.container.demux(self.stream)

        # Need 960 samples * 2 bytes = 1920 bytes per frame
        bytes_needed = self.SAMPLES_PER_FRAME * 2

        # Fill buffer until we have enough
        while len(self._buffer) < bytes_needed:
            try:
                packet = next(self._packets)
                for frame in packet.decode():
                    resampled = self.resampler.resample(frame)
                    for r_frame in resampled:
                        self._buffer += bytes(r_frame.planes[0])
            except StopIteration:
                # File ended, loop
                self.container.seek(0)
                self._packets = self.container.demux(self.stream)
                self._frame_count = 0
                self._start_time = time.time()
                # Pad with silence if needed
                if len(self._buffer) < bytes_needed:
                    self._buffer += b'\x00' * (bytes_needed - len(self._buffer))
                break

        # Extract exactly 960 samples
        frame_data = self._buffer[:bytes_needed]
        self._buffer = self._buffer[bytes_needed:]

        # Create audio frame
        samples = np.frombuffer(frame_data, dtype=np.int16)
        audio_frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1),
            format='s16',
            layout='mono'
        )
        audio_frame.sample_rate = 48000
        audio_frame.pts = self._frame_count * self.SAMPLES_PER_FRAME
        audio_frame.time_base = fractions.Fraction(1, 48000)

        self._frame_count += 1

        # Pace the frames (20ms each)
        elapsed = time.time() - self._start_time
        expected = self._frame_count * 0.02
        if expected > elapsed:
            await asyncio.sleep(expected - elapsed)

        return audio_frame


async def send_audio_to_speaker(audio_path, stream_name=CAMERA_STREAM):
    """Send audio file to Tapo speaker via go2rtc WebRTC."""

    print(f"\nConnecting to go2rtc...")
    print(f"Stream: {stream_name}")
    print(f"Audio: {audio_path}")

    # Create WebRTC connection
    pc = RTCPeerConnection()

    # Add audio track
    audio_track = AudioFileTrack(audio_path)
    pc.addTrack(audio_track)

    # Create offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    print("Sending WebRTC offer to go2rtc...")

    # Send offer to go2rtc
    async with aiohttp.ClientSession() as session:
        # go2rtc WebRTC API endpoint
        url = f"{GO2RTC_URL}/api/webrtc?src={stream_name}"

        async with session.post(
            url,
            data=pc.localDescription.sdp,
            headers={"Content-Type": "application/sdp"}
        ) as resp:
            if resp.status not in (200, 201):
                text = await resp.text()
                print(f"Error: {resp.status} - {text}")
                return False

            # Get answer
            answer_sdp = await resp.text()
            answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
            await pc.setRemoteDescription(answer)

            print("WebRTC connected!")
            print(f"Playing audio to Tapo speaker...")

    # Wait for audio to play (or until interrupted)
    try:
        # Get audio duration
        container = av.open(audio_path)
        duration = float(container.duration) / av.time_base if container.duration else 30
        container.close()

        print(f"Duration: {duration:.1f} seconds")
        print("Press Ctrl+C to stop\n")

        await asyncio.sleep(duration + 1)

    except KeyboardInterrupt:
        print("\nStopped!")

    finally:
        await pc.close()

    return True


async def test_connection():
    """Test if go2rtc is running and has streams."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{GO2RTC_URL}/api/streams") as resp:
                if resp.status == 200:
                    streams = await resp.json()
                    print(f"go2rtc streams: {list(streams.keys())}")
                    return list(streams.keys())
    except Exception as e:
        print(f"go2rtc not running: {e}")
    return []


def main():
    print("=" * 60)
    print("  TAPO SPEAKER PLUGIN")
    print("  Send audio files to camera speaker!")
    print("=" * 60)

    # Get audio file
    if len(sys.argv) >= 2:
        audio_path = sys.argv[1]
    else:
        audio_path = r"C:\Users\Cheng Jit Giam\Downloads\Mambu OMatsuri mambu - (320 Kbps).mp3"

    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return

    print(f"\nAudio file: {audio_path}")

    # Test connection
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    streams = loop.run_until_complete(test_connection())

    if not streams:
        print("\ngo2rtc not running!")
        print("Start it first: python setup_go2rtc.py .129")
        return

    # Use first available stream
    stream_name = streams[0] if streams else CAMERA_STREAM
    print(f"Using stream: {stream_name}")

    # Send audio
    try:
        loop.run_until_complete(send_audio_to_speaker(audio_path, stream_name))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
