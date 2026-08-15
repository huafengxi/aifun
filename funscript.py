#!/usr/bin/env python3
"""
funscript.py — Generate funscript from siko timestamps.

Reads siko.py TSV output (stdin) and generates a funscript JSON file.

Usage:
    ./siko.py a.mp3 | ./funscript.py -o output.funscript
    ./siko.py a.mp3 | ./funscript.py --freq 6.0 --range 80
"""

import argparse
import json
import sys
import math


def parse_tsv(lines):
    """Parse siko.py TSV output: start_ts  duration"""
    timings = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                start = float(parts[0])
                duration = float(parts[1])
                timings.append((start, duration))
            except ValueError:
                continue
    return timings


def generate_funscript(
    timings,
    fps=60,
    freq=5.0,       # oscillation frequency (Hz)
    amplitude=90,   # max amplitude (0-100)
    phase=0,        # initial phase offset
    attack=0.1,     # attack time (seconds)
    release=0.15,   # release time (seconds)
):
    """Generate funscript actions from siko timings."""
    actions = []
    action_id = 0

    for start, duration in timings:
        end = start + duration
        t = start
        dt = 1.0 / fps

        # Ease-in (attack)
        attack_frames = max(1, int(attack / dt))
        release_frames = max(1, int(release / dt))

        # Duration in frames
        total_frames = max(1, int(duration / dt))

        for i in range(total_frames):
            t = start + i * dt
            if t > end:
                break

            # Oscillation: sin wave, 0..1
            raw = 0.5 + 0.5 * math.sin(2 * math.pi * freq * t + phase)

            # Apply attack/release envelope
            env = 1.0
            if i < attack_frames:
                env = i / attack_frames
            elif i > total_frames - release_frames:
                remaining = total_frames - i
                env = remaining / release_frames
            env = max(0.0, min(1.0, env))

            # Scale to amplitude range
            pos = int(round(raw * amplitude * env))
            pos = max(0, min(99, pos))

            # Round to integer frame
            at = round(t * fps)
            actions.append({
                "at": at,
                "pos": pos,
            })

            if len(actions) > 100000:
                print("Warning: Too many actions, truncating", file=sys.stderr)
                break

    # Sort by time
    actions.sort(key=lambda x: x["at"])

    # Remove duplicates (same timestamp)
    deduped = []
    for a in actions:
        if not deduped or deduped[-1]["at"] != a["at"]:
            deduped.append(a)

    return {
        "version": "1.0",
        "inverted": False,
        "range": amplitude,
        "actions": deduped,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate funscript from siko timestamps."
    )
    parser.add_argument("-o", "--output", default="output.funscript",
                        help="Output funscript file (default: output.funscript)")
    parser.add_argument("--freq", type=float, default=5.0,
                        help="Oscillation frequency in Hz (default: 5.0)")
    parser.add_argument("--range", type=int, default=80,
                        help="Amplitude range 0-100 (default: 80)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Frames per second (default: 60)")
    parser.add_argument("--attack", type=float, default=0.1,
                        help="Attack time in seconds (default: 0.1)")
    parser.add_argument("--release", type=float, default=0.15,
                        help="Release time in seconds (default: 0.15)")

    args = parser.parse_args()

    # Read timings from stdin
    timings = parse_tsv(sys.stdin)

    if not timings:
        print("No timings found on stdin", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(timings)} timing(s)", file=sys.stderr)

    funscript = generate_funscript(
        timings,
        fps=args.fps,
        freq=args.freq,
        amplitude=args.range,
        attack=args.attack,
        release=args.release,
    )

    with open(args.output, "w") as f:
        json.dump(funscript, f, indent=2)

    print(f"Written {len(funscript['actions'])} actions to {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()