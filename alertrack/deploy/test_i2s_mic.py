#!/usr/bin/env python3
"""Quick INMP441 I2S capture test — run on the Pi AFTER the setup steps.

Captures 3 s from the ALSA default (the I2S mic) at the model's 44.1 kHz mono and
reports level. A working mic gives peak/RMS that clearly rise when you clap or
speak; near-zero on silence. Writes test_i2s.wav so you can listen back.

    python3 alertrack/deploy/test_i2s_mic.py
    python3 alertrack/deploy/test_i2s_mic.py --device googlevoicehat --seconds 5
"""
import argparse
import wave
import numpy as np
import sounddevice as sd

SR = 44_100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None,
                    help="None=ALSA default; or index/name e.g. 'googlevoicehat'")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", default="test_i2s.wav")
    args = ap.parse_args()

    dev = args.device
    if dev is not None and dev.isdigit():
        dev = int(dev)

    print("Input devices:")
    print(sd.query_devices())
    print(f"\nRecording {args.seconds:.0f}s @ {SR} Hz mono from device={dev!r} ... "
          "(make some noise — clap / speak)")

    audio = sd.rec(int(args.seconds * SR), samplerate=SR, channels=1,
                   dtype="float32", device=dev)
    sd.wait()
    audio = audio.flatten()

    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"\n  samples : {audio.size}")
    print(f"  peak    : {peak:.5f}")
    print(f"  rms     : {rms:.5f}")
    if peak < 1e-4:
        print("  ⚠️  Essentially silent — check wiring, L/R pin, and that the overlay loaded.")
    else:
        print("  ✅ Signal present. (Absolute level is low on INMP441 — EBU-R128 "
              "normalisation in the daemon compensates.)")

    pcm = np.clip(audio, -1, 1)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())
    print(f"  wrote {args.out}  (aplay {args.out} to listen)")


if __name__ == "__main__":
    main()
