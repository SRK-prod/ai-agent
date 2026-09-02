#!/usr/bin/env python3
"""Play whatever is going into the virtual cable back out through the real speakers.

This is a userspace replacement for Windows' "Listen to this device" checkbox.

WHY: the meeting app's speaker is set to CABLE Input so the copilot can transcribe the
interviewer -- which means that audio no longer reaches the speakers and the candidate
cannot hear the call. Windows' own monitor solves it, but the setting lives under an HKLM
key owned by SYSTEM: it cannot be written even from an elevated shell without taking
ownership of system registry keys, which is not a reasonable thing to do for a checkbox.

Copying the samples ourselves needs no admin, no registry and no system settings, and it
is trivially reversible -- close the window and it stops.

    .venv\\Scripts\\python.exe scripts\\audio_monitor.py
    .venv\\Scripts\\python.exe scripts\\audio_monitor.py --list
    .venv\\Scripts\\python.exe scripts\\audio_monitor.py --gain 1.5 --blocksize 512

Leave it running for the whole call. Ctrl+C stops it.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import sounddevice as sd

# WASAPI gives the lowest latency of the host APIs Windows exposes here; MME is the
# fallback because it is always present even when a device exposes no WASAPI endpoint.
_PREFERRED_HOSTAPIS = ("Windows WASAPI", "Windows DirectSound", "MME")


def _pick(name_fragment: str, want_input: bool) -> int:
    """Find a device by name, preferring the lowest-latency host API available for it."""
    candidates = []
    for idx, dev in enumerate(sd.query_devices()):
        if name_fragment.lower() not in dev["name"].lower():
            continue
        channels = dev["max_input_channels"] if want_input else dev["max_output_channels"]
        if channels < 1:
            continue
        api = sd.query_hostapis(dev["hostapi"])["name"]
        rank = _PREFERRED_HOSTAPIS.index(api) if api in _PREFERRED_HOSTAPIS else 99
        candidates.append((rank, idx, dev["name"], api, channels))
    if not candidates:
        raise SystemExit(
            f"No {'input' if want_input else 'output'} device matching {name_fragment!r}."
        )
    candidates.sort()
    rank, idx, name, api, channels = candidates[0]
    print(f"  {'in ' if want_input else 'out'}: [{idx}] {name.strip()}  ({api}, {channels}ch)")
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="CABLE Output", help="device to listen TO")
    ap.add_argument("--target", default="Speaker/HP", help="device to play OUT of")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--samplerate", type=int, default=48000)
    # 1024 frames at 48kHz is ~21ms per block. Lower is tighter but risks dropouts on a
    # busy 2-core machine; raise this first if the audio crackles during a call.
    ap.add_argument("--blocksize", type=int, default=1024)
    ap.add_argument("--list", action="store_true", help="list devices and exit")
    args = ap.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    print("audio monitor -- cable -> speakers")
    src = _pick(args.source, want_input=True)
    dst = _pick(args.target, want_input=False)

    # Mirror the cable's audio to the speakers. The meeting app writes stereo into the
    # cable, so read 2 channels and write 2 -- the copilot separately opens the same
    # device in mono for transcription, which is fine: WASAPI shared mode allows it.
    def callback(indata, outdata, frames, t, status):
        if status:
            # Over/underruns are worth seeing but must never stop the stream mid-interview.
            print(f"  [{time.strftime('%H:%M:%S')}] {status}", file=sys.stderr)
        np.multiply(indata, args.gain, out=outdata, casting="unsafe")

    try:
        with sd.Stream(
            device=(src, dst),
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            dtype="float32",
            channels=2,
            callback=callback,
        ):
            latency_ms = 1000 * args.blocksize / args.samplerate
            print(f"\nMONITORING -- ~{latency_ms:.0f}ms block latency. Ctrl+C to stop.")
            print("You should now hear the meeting through your speakers.")
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopped.")
    except Exception as exc:  # surface the real cause, do not dump a traceback mid-call
        raise SystemExit(f"\nmonitor failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
