#!/usr/bin/env python3
"""
siko.py — Detect 西口 (siko) instructions in Japanese ASMR audio.

Uses Qwen3-ASR-1.7B for speech recognition with timestamp alignment.

Usage:
    ./siko.py <audio_file> [--model MODEL] [--device DEVICE] [--compute COMPUTE_TYPE]

Output:
    <start_ts> <duration>    # for each detected siko instruction
"""

import argparse
import sys
import os
import re
import json
from typing import List, Optional, Tuple


# Patterns to detect siko-related instructions
SIKO_PATTERNS = [
    re.compile(r'しこ', re.IGNORECASE),
    re.compile(r'シコ', re.IGNORECASE),
    re.compile(r'西口'),
    re.compile(r'siko', re.IGNORECASE),
    re.compile(r'shico', re.IGNORECASE),
    re.compile(r'shikoko', re.IGNORECASE),
    re.compile(r'しこしこ', re.IGNORECASE),
    re.compile(r'シコシコ', re.IGNORECASE),
]


def format_ts(seconds: float) -> str:
    """Format seconds to SRT-style timestamp: HH:MM:SS.mmm"""
    return f"{seconds:.3f}"


def format_srt_ts(seconds: float) -> str:
    """Format seconds to SRT-style timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')


def detect_siko_in_text(text: str) -> List[Tuple[str, int, int]]:
    """
    Detect siko patterns in text and return (pattern_text, start_pos, end_pos) for each match.
    """
    results = []
    for pattern in SIKO_PATTERNS:
        for match in pattern.finditer(text):
            results.append((match.group(), match.start(), match.end()))
    # Deduplicate overlapping matches (keep the first/longest)
    results.sort(key=lambda x: x[1])
    deduped = []
    for r in results:
        if not deduped or r[1] >= deduped[-1][2]:
            deduped.append(r)
        elif r[2] > deduped[-1][2]:
            deduped[-1] = r
    return deduped


def detect_siko_from_timestamps(
    text: str,
    time_stamps: List,
) -> List[Tuple[float, float, str]]:
    """
    Detect siko instructions using word-level timestamps from Qwen3-ASR + ForcedAligner.

    time_stamps: list of objects with .text, .start_time, .end_time attributes
    """
    results = []
    # Build a character-to-timestamp map
    char_to_ts = {}  # char_index -> (start_time, end_time)
    for ts in time_stamps:
        # ts.text is the word/character, ts.start_time, ts.end_time
        # We need to find the position in the full text
        # But since we have the aligned timestamps, we can directly check each word
        word_text = ts.text
        for pattern in SIKO_PATTERNS:
            if pattern.search(word_text):
                results.append((
                    ts.start_time,
                    ts.end_time - ts.start_time,
                    word_text,
                ))
                break

    return results


def detect_siko_from_segments(segments: List) -> List[Tuple[float, float, str]]:
    """
    Fallback: detect siko instructions in transcribed segments (without aligner).
    segments: list of objects with .text, .start_time, .end_time
    """
    results = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        for pattern in SIKO_PATTERNS:
            if pattern.search(text):
                results.append((
                    seg.start_time,
                    seg.end_time - seg.start_time,
                    text,
                ))
                break
    return results


def resolve_model(
    model_id: str,
    skip_download: bool = False,
) -> str:
    """
    Resolve a model ID to a local path.

    If MODEL_DOWNLOAD_SOURCE env var is set to 'modelscope' or 'auto',
    tries to download from ModelScope first. Falls back to HuggingFace.

    Returns the model ID (for HF hub) or local path (for ModelScope download).
    """
    if skip_download:
        return model_id

    source = os.environ.get("MODEL_DOWNLOAD_SOURCE", "auto").lower()
    if source == "huggingface":
        # HuggingFace is the default for qwen-asr, just return the model ID
        return model_id

    # Try to use download_model.py logic
    try:
        from download_model import resolve_model_path

        cache_dir = (
            os.environ.get("MODELSCOPE_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
        )
        local_path = resolve_model_path(model_id, source=source, cache_dir=cache_dir)
        return local_path
    except ImportError:
        # download_model.py not available, try inline modelscope
        pass
    except Exception as e:
        print(f"Warning: Model download failed: {e}", file=sys.stderr)
        print(f"Falling back to model ID: {model_id}", file=sys.stderr)

    return model_id


def main():
    parser = argparse.ArgumentParser(
        description="Detect 西口 (siko) instructions in Japanese ASMR audio."
    )
    parser.add_argument("audio_file", help="Path to the audio file (MP3, WAV, etc.)")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-ASR-1.7B",
        help="Qwen3-ASR model name or path (default: Qwen/Qwen3-ASR-1.7B)"
    )
    parser.add_argument(
        "--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Forced aligner model for timestamp alignment (default: Qwen/Qwen3-ForcedAligner-0.6B)"
    )
    parser.add_argument(
        "--no-aligner", action="store_true",
        help="Disable forced aligner (no timestamps, segment-level only)"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device to run on: 'auto', 'cuda:0', 'cpu'"
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        help="Compute dtype: 'bfloat16', 'float16', 'float32'"
    )
    parser.add_argument(
        "--language", default="ja",
        help="Language code (default: ja for Japanese)"
    )
    parser.add_argument(
        "--output", choices=["tsv", "srt", "json"], default="tsv",
        help="Output format (default: tsv)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum new tokens for generation (default: 256)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Max inference batch size (default: 1)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: File not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    # Import qwen_asr
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError:
        print(
            "Error: qwen-asr is required.\n"
            "Install it with: pip install -U qwen-asr",
            file=sys.stderr
        )
        sys.exit(1)

    import torch

    # Determine device
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dtype = getattr(torch, args.dtype, torch.bfloat16)
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
        print(f"Info: CPU detected, using dtype=float32", file=sys.stderr)

    # Resolve model path — try ModelScope first if configured
    model_path = resolve_model(args.model, args.no_aligner)
    aligner_path = resolve_model(args.aligner, args.no_aligner) if not args.no_aligner else None

    print(f"Loading model '{model_path}' on {device} (dtype={dtype})...",
          file=sys.stderr)

    # Initialize model with optional forced aligner
    model_kwargs = dict(
        model=model_path,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    if not args.no_aligner and aligner_path:
        model_kwargs["forced_aligner"] = aligner_path
        model_kwargs["forced_aligner_kwargs"] = dict(
            dtype=dtype,
            device_map=device,
        )

    model = Qwen3ASRModel.from_pretrained(**model_kwargs)

    print(f"Transcribing {args.audio_file}...", file=sys.stderr)

    # Run transcription
    # The model can be called with a single audio or a list
    language = args.language if args.language else None
    results = model.transcribe(
        audio=args.audio_file,
        language=language,
        return_time_stamps=not args.no_aligner,
    )

    # results is a list (even for single audio)
    if not isinstance(results, list):
        results = [results]

    result = results[0]
    print(f"Detected language: {result.language}", file=sys.stderr)

    # Detect siko instructions
    if not args.no_aligner and hasattr(result, 'time_stamps') and result.time_stamps:
        # Use word-level timestamps from forced aligner
        siko_results = detect_siko_from_timestamps(result.text, result.time_stamps)
    else:
        # Fallback: use segment-level or just full text
        if hasattr(result, 'segments') and result.segments:
            siko_results = detect_siko_from_segments(result.segments)
        else:
            # Full text search without timestamps
            matches = detect_siko_in_text(result.text)
            # Without timestamps, we can't output meaningful timestamps
            siko_results = [(0.0, 0.0, text) for text, _, _ in matches]

    print(f"Found {len(siko_results)} siko instruction(s)", file=sys.stderr)
    if siko_results:
        print(f"{'Timestamp':>12} {'Duration':>10} {'Text':<40}", file=sys.stderr)
        print("-" * 64, file=sys.stderr)
        for ts, dur, text in siko_results:
            print(f"  {format_ts(ts):>10}  {dur:>8.3f}  {text.strip()[:40]}",
                  file=sys.stderr)

    # Output in requested format
    if args.output == "tsv":
        for ts, dur, _ in siko_results:
            print(f"{format_ts(ts)}\t{dur:.3f}")
    elif args.output == "srt":
        for i, (ts, dur, text) in enumerate(siko_results, 1):
            end_ts = ts + dur
            print(f"{i}")
            print(f"{format_srt_ts(ts)} --> {format_srt_ts(end_ts)}")
            print(f"{text.strip()}")
            print()
    elif args.output == "json":
        output = []
        for ts, dur, text in siko_results:
            output.append({
                "start": round(ts, 3),
                "duration": round(dur, 3),
                "end": round(ts + dur, 3),
                "text": text.strip(),
            })
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()