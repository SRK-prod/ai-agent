#!/usr/bin/env python3
"""Play a WAV out of the DEFAULT OUTPUT device (which should be CABLE Input) so the
running pipeline captures it exactly as it would capture a live call. Tests the whole
chain -- capture -> VAD -> diarization -> STT -> question detection -> Claude -> overlay --
without needing a real meeting.

    .venv\\Scripts\\python.exe scripts\\play_into_cable.py logs\\sample_client.wav --start 5 --seconds 30
"""

from __future__ import annotations

import argparse
import wave

import numpy as np
import sounddevice as sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--gain", type=float, default=1.0, help="volume multiplier")
    args = ap.parse_args()

    with wave.open(args.wav, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)

    s = int(args.start * sr)
    e = s + int(args.seconds * sr)
    clip = np.clip(a[s:e] * args.gain, -1.0, 1.0)

    # Target CABLE Input explicitly rather than the default output. Once the machine is set
    # up correctly the Windows default is the real speakers (so the user can hear normally)
    # and only the meeting app is pointed at the cable -- so playing to "default" would go to
    # the speakers and test nothing.
    device = None
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and "CABLE Input" in d["name"]:
            device = i
            break
    if device is None:
        raise SystemExit("no 'CABLE Input' output device found -- is VB-Cable installed?")

    print(f"playing {len(clip) / sr:.1f}s into: {sd.query_devices(device)['name']}")
    print(f"clip level: peak={np.abs(clip).max():.3f} rms={np.sqrt(np.mean(clip**2)):.4f}")
    sd.play(clip, sr, device=device)
    sd.wait()
    print("done -- watch the overlay")


if __name__ == "__main__":
    main()
