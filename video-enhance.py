#!/usr/bin/env python3
"""
video-enhance.py — Optimize video quality and re-encode to H.265 (HEVC).

Pipeline: denoise (hqdn3d) -> sharpen (cas) -> optional upscale (lanczos)
          -> encode with hevc_nvenc (GPU) or libx265 (CPU).

Usage:
    ./video-enhance.py input.mp4
    ./video-enhance.py input.mp4 -o output.mp4 --scale 2 --cq 20
    ./video-enhance.py input.mp4 --encoder cpu --dry-run

Options:
    -o, --output FILE   Output file (default: <input>.enhanced.mp4)
    --scale N           Upscale factor, e.g. 1.5 or 2 (default: 1 = native)
    --denoise F         Denoise strength 0.0-1.0 (default: 0.3, 0 = off)
    --sharpen F         Sharpen amount 0.0-1.0 (default: 0 = off; cas costs
                        ~1/3 of runtime, use --sharpen 0.4 to restore)
    --cq N              Quality target: NVENC -cq / x265 -crf (default: 22)
    --encoder MODE      auto | gpu | cpu (default: auto)
    --force             Overwrite existing output file
    --dry-run           Print the ffmpeg command without running it
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def run(cmd, capture=True):
    """Run a command, return (rc, stdout, stderr)."""
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def probe(path):
    """Return first video stream info via ffprobe."""
    rc, out, err = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,codec_name",
        "-show_entries", "format=duration,size",
        "-of", "json", path,
    ])
    if rc != 0:
        die(f"ffprobe failed on {path}: {err.strip()}")
    info = json.loads(out)
    streams = info.get("streams") or []
    if not streams:
        die(f"no video stream found in {path}")
    return info, streams[0]


# Audio codecs that can be stream-copied into an MP4 container.
# Anything else (wma*/wmapro, vorbis, opus, flac, pcm_*, dts, truehd, ...)
# gets transcoded to AAC.
MP4_COMPAT_AUDIO = {"aac", "mp3", "mp2", "ac3", "eac3", "alac"}
AAC_BITRATE = "192k"


def probe_audio_streams(path):
    """Return list of audio streams (index, codec_name) via ffprobe."""
    rc, out, err = run([
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index,codec_name",
        "-of", "json", path,
    ])
    if rc != 0:
        die(f"ffprobe (audio) failed on {path}: {err.strip()}")
    return json.loads(out).get("streams") or []


def audio_plan(path):
    """Decide per-audio-stream handling for the MP4 output.

    Returns (opts, summary) where opts is the ffmpeg option list and
    summary is a printable list like ['a:0 aac -> copy', 'a:1 wmapro -> aac 192k'].
    """
    opts, summary = [], []
    for n, st in enumerate(probe_audio_streams(path)):
        codec = (st.get("codec_name") or "unknown").lower()
        if codec in MP4_COMPAT_AUDIO:
            opts += [f"-c:a:{n}", "copy"]
            summary.append(f"a:{n} {codec} -> copy")
        else:
            opts += [f"-c:a:{n}", "aac", f"-b:a:{n}", AAC_BITRATE]
            summary.append(f"a:{n} {codec} -> aac {AAC_BITRATE} (not mp4-compatible)")
    return opts, summary


def nvenc_available():
    """Quick check that hevc_nvenc actually works on this GPU."""
    rc, _, err = run([
        "ffmpeg", "-hide_banner", "-v", "error",
        "-f", "lavfi", "-i", "nullsrc=s=512x512:d=0.1",
        "-c:v", "hevc_nvenc", "-frames:v", "1",
        "-f", "null", "-",
    ])
    if rc != 0:
        print(f"note: hevc_nvenc not usable ({err.strip()[:120]})", file=sys.stderr)
        return False
    return True


def build_filters(args):
    """Build the -vf filter chain."""
    vf = []
    if args.denoise > 0:
        # hqdn3d: map 0..1 -> gentle..strong
        s = args.denoise
        vf.append(f"hqdn3d=luma_spatial={3 + 9 * s:.1f}:chroma_spatial={2 + 6 * s:.1f}"
                  f":luma_tmp={4 + 8 * s:.1f}:chroma_tmp={3 + 6 * s:.1f}")
    if args.sharpen > 0:
        # cas: contrast adaptive sharpening, strength 0..1
        vf.append(f"cas=strength={min(args.sharpen, 1.0):.2f}")
    if args.scale != 1.0:
        # keep dimensions even (required by many encoders)
        vf.append(f"scale=trunc(iw*{args.scale}/2)*2:trunc(ih*{args.scale}/2)*2:flags=lanczos")
    return vf


def build_cmd(args, vf, encoder, audio_opts):
    """Assemble the full ffmpeg command."""
    cmd = ["ffmpeg", "-hide_banner", "-y" if args.force else "-n", "-i", args.input]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if encoder == "gpu":
        cmd += ["-c:v", "hevc_nvenc", "-preset", "p5",
                "-rc", "vbr", "-cq", str(args.cq), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx265", "-crf", str(args.cq), "-preset", "medium"]
    cmd += ["-tag:v", "hvc1",        # Apple/QuickTime compatibility
            *audio_opts,             # copy mp4-compatible audio, else transcode to AAC
            "-map_metadata", "0",
            "-movflags", "+faststart",
            args.output]
    return cmd


def main():
    ap = argparse.ArgumentParser(
        description="Optimize video quality and re-encode to H.265 (HEVC).")
    ap.add_argument("input", help="input video file")
    ap.add_argument("-o", "--output", help="output file (default: <input>.enhanced.mp4)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="upscale factor, e.g. 2 for 2x (default: 1)")
    ap.add_argument("--denoise", type=float, default=0.3,
                    help="denoise strength 0.0-1.0 (default: 0.3)")
    ap.add_argument("--sharpen", type=float, default=0.0,
                    help="sharpen amount 0.0-1.0 (default: 0 = off; "
                         "use 0.4 to restore the legacy default)")
    ap.add_argument("--cq", type=int, default=22,
                    help="quality target: NVENC -cq / x265 -crf, lower = better (default: 22)")
    ap.add_argument("--encoder", choices=["auto", "gpu", "cpu"], default="auto",
                    help="encoder selection (default: auto)")
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    ap.add_argument("--dry-run", action="store_true", help="print command and exit")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            die(f"{tool} not found in PATH")
    if not os.path.isfile(args.input):
        die(f"input not found: {args.input}")
    if args.scale < 1.0:
        die("--scale must be >= 1.0 (downscaling is not 'enhancement')")

    if not args.output:
        base, _ = os.path.splitext(args.input)
        args.output = f"{base}.enhanced.mp4"
    if os.path.exists(args.output) and not args.force and not args.dry_run:
        die(f"output exists: {args.output} (use --force to overwrite)")

    # Input info
    info, vs = probe(args.input)
    dur = info.get("format", {}).get("duration")
    dur_s = f", {float(dur):.1f}s" if dur else ""
    print(f"input : {args.input} ({vs['width']}x{vs['height']}, "
          f"{vs.get('codec_name', '?')}, {vs.get('r_frame_rate', '?')} fps{dur_s})")

    # Encoder selection
    if args.encoder == "gpu":
        encoder = "gpu"
    elif args.encoder == "cpu":
        encoder = "cpu"
    else:
        encoder = "gpu" if nvenc_available() else "cpu"
    print(f"encode: {'hevc_nvenc (NVENC GPU)' if encoder == 'gpu' else 'libx265 (CPU)'} "
          f"cq={args.cq}")

    vf = build_filters(args)
    if vf:
        print(f"filters: {' -> '.join(vf)}")

    audio_opts, audio_summary = audio_plan(args.input)
    print(f"audio : {', '.join(audio_summary) if audio_summary else 'no audio stream'}")

    cmd = build_cmd(args, vf, encoder, audio_opts)
    if args.dry_run:
        print("cmd   :", " ".join(cmd))
        return

    print(f"output: {args.output}")
    # Stream ffmpeg stderr (progress) to the terminal.
    p = subprocess.run(cmd, stderr=None)
    if p.returncode != 0:
        # Common case: NVENC died mid-encode. Retry once on CPU if we used GPU.
        if encoder == "gpu":
            print("note: GPU encode failed, retrying with libx265 (CPU)...", file=sys.stderr)
            if os.path.exists(args.output):
                os.remove(args.output)  # drop partial output from the GPU attempt
            cmd = build_cmd(args, vf, "cpu", audio_opts)
            p = subprocess.run(cmd, stderr=None)
        if p.returncode != 0:
            sys.exit(p.returncode)

    size_in = os.path.getsize(args.input)
    size_out = os.path.getsize(args.output)
    print(f"done  : {size_in / 1e6:.1f} MB -> {size_out / 1e6:.1f} MB "
          f"({size_out / size_in * 100:.0f}%)")


if __name__ == "__main__":
    main()
