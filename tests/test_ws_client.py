from __future__ import annotations

import asyncio
import json
import struct
import sys

import numpy as np


def _generate_test_pcm(duration_sec: float = 2.0, sample_rate: int = 16000) -> bytes:
    samples = int(sample_rate * duration_sec)
    t = np.arange(samples) / sample_rate
    signal = (
        np.sin(2 * np.pi * 440 * t) * 0.3
        + np.sin(2 * np.pi * 660 * t) * 0.15
        + np.random.randn(samples) * 0.01
    )
    signal = np.clip(signal * 0.8, -1.0, 1.0)
    return (signal * 32767).astype(np.int16).tobytes()


async def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8001"
    api_key = sys.argv[2] if len(sys.argv) > 2 else "ak_aiih_xxx"

    uri = f"ws://{host}/v1/audio/transcriptions/stream?api_key={api_key}"

    try:
        import websockets
    except ImportError:
        print("Missing websockets library. Install: pip install websockets")
        sys.exit(1)

    print(f"Connecting to {uri} ...")
    async with websockets.connect(uri) as ws:
        print("Connected! Streaming audio...")

        pcm = _generate_test_pcm(3.0)
        chunk_size = 16000  # 1 second chunks

        for i in range(0, len(pcm), chunk_size):
            chunk = pcm[i : i + chunk_size]
            await ws.send(chunk)
            print(f"  Sent {len(chunk)} bytes ({len(chunk)//320} frames)")
            await asyncio.sleep(0.5)

        await ws.send(json.dumps({"type": "flush"}).encode())

        print("Waiting for transcripts...")
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(msg)
                print(f"  << {json.dumps(data, ensure_ascii=False)}")
                if data.get("type") == "transcript":
                    print(f"  => Transcript: {data['text']}")
        except asyncio.TimeoutError:
            print("  (timeout - no more messages)")
        except websockets.exceptions.ConnectionClosed:
            print("  (connection closed)")

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
