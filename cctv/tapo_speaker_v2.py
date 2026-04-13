#!/usr/bin/env python3
"""
Tapo Speaker v2 - Using G.711 codec (PCMU/8kHz)
"""

import asyncio
import aiohttp
import av
import time
import os
import sys
import fractions
import numpy as np

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.codecs import get_encoder, get_decoder
from aiortc.mediastreams import AudioStreamTrack

GO2RTC_URL = "http://localhost:1984"
CAMERA_STREAM = "tapo_camera"


class G711AudioTrack(MediaStreamTrack):
    """
    Audio track that outputs G.711 compatible frames (8kHz mono).
    """
    kind = "audio"

    # G.711 uses 8kHz, 160 samples per 20ms frame
    SAMPLE_RATE = 8000
    SAMPLES_PER_FRAME = 160  # 20ms at 8kHz

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._started = False
        self._container = None
        self._resampler = None
        self._buffer = b''
        self._packets = None
        self._frame_count = 0
        self._start_time = None

    async def recv(self):
        if not self._started:
            self._started = True
            self._start_time = time.time()

            # Open file and setup resampler for 8kHz mono
            self._container = av.open(self.file_path)
            self._stream = self._container.streams.audio[0]
            self._resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=self.SAMPLE_RATE
            )
            self._packets = self._container.demux(self._stream)

        # Need 160 samples * 2 bytes = 320 bytes per frame
        bytes_needed = self.SAMPLES_PER_FRAME * 2

        # Fill buffer
        while len(self._buffer) < bytes_needed:
            try:
                packet = next(self._packets)
                for frame in packet.decode():
                    resampled = self._resampler.resample(frame)
                    for r_frame in resampled:
                        self._buffer += bytes(r_frame.planes[0])
            except StopIteration:
                # Loop the audio
                self._container.seek(0)
                self._packets = self._container.demux(self._stream)
                if len(self._buffer) < bytes_needed:
                    self._buffer += b'\x00' * (bytes_needed - len(self._buffer))
                break

        # Extract frame data
        frame_data = self._buffer[:bytes_needed]
        self._buffer = self._buffer[bytes_needed:]

        # Create av.AudioFrame
        samples = np.frombuffer(frame_data, dtype=np.int16)
        audio_frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1),
            format='s16',
            layout='mono'
        )
        audio_frame.sample_rate = self.SAMPLE_RATE
        audio_frame.pts = self._frame_count * self.SAMPLES_PER_FRAME
        audio_frame.time_base = fractions.Fraction(1, self.SAMPLE_RATE)

        self._frame_count += 1

        # Pace at 20ms per frame
        elapsed = time.time() - self._start_time
        expected = self._frame_count * 0.02
        if expected > elapsed:
            await asyncio.sleep(expected - elapsed)

        return audio_frame


async def send_audio(audio_path, stream_name):
    """Send audio to Tapo speaker via WebRTC."""

    print(f"Connecting to go2rtc...")
    print(f"Stream: {stream_name}")
    print(f"Audio: {audio_path}")

    # Create peer connection
    pc = RTCPeerConnection()

    # Add audio track
    audio_track = G711AudioTrack(audio_path)
    pc.addTrack(audio_track)

    # Create and set local offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    print("Sending WebRTC offer...")

    # Send to go2rtc
    async with aiohttp.ClientSession() as session:
        url = f"{GO2RTC_URL}/api/webrtc?src={stream_name}"

        async with session.post(
            url,
            data=pc.localDescription.sdp,
            headers={"Content-Type": "application/sdp"}
        ) as resp:
            if resp.status not in (200, 201):
                text = await resp.text()
                print(f"Error: {resp.status} - {text}")
                return

            answer_sdp = await resp.text()
            print(f"Got answer, setting remote description...")

            answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
            await pc.setRemoteDescription(answer)

    print("WebRTC connected!")
    print("Playing audio to speaker...")

    # Get duration
    container = av.open(audio_path)
    duration = container.duration / 1000000 if container.duration else 10
    container.close()

    print(f"Duration: {duration:.1f}s - Press Ctrl+C to stop\n")

    try:
        await asyncio.sleep(duration + 2)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        await pc.close()


async def test_streams():
    """List available streams."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{GO2RTC_URL}/api/streams") as resp:
            if resp.status == 200:
                streams = await resp.json()
                return list(streams.keys())
    return []


def main():
    print("=" * 50)
    print("  TAPO SPEAKER V2 (G.711 codec)")
    print("=" * 50)

    # Audio file
    if len(sys.argv) >= 2:
        audio_path = sys.argv[1]
    else:
        audio_path = r"C:\Users\Cheng Jit Giam\Downloads\Mambu OMatsuri mambu - (320 Kbps).mp3"

    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Check streams
    streams = loop.run_until_complete(test_streams())
    if not streams:
        print("go2rtc not running!")
        return

    print(f"Streams: {streams}")
    stream = streams[0]

    # Send audio
    loop.run_until_complete(send_audio(audio_path, stream))
    loop.close()


if __name__ == "__main__":
    main()
