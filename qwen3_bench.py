#!/usr/bin/env python3
"""
qwen3_bench.py — Single-stream (single connection) decode perf test.

Measures decode speed against the Qwen3 OpenAI-compatible server started by
`qwen3_serve.py` (or any vLLM OpenAI endpoint). One request at a time, in
sequence — no concurrency.

Usage:
    ./qwen3_bench.py                     # decode throughput + TTFT, 3 runs
    ./qwen3_bench.py --max-tokens 1024 --runs 5
    ./qwen3_bench.py --no-stream         # throughput only (no TTFT)

Environment variables:
    QWEN3_BENCH_URL     OpenAI base URL   (default: http://127.0.0.1:8000)
    QWEN3_BENCH_MODEL   Model name        (default: qwen3)
"""

import argparse
import http.client
import json
import os
import time
from urllib.parse import urlparse


def _connect(base_url: str, timeout: int = 600):
    u = urlparse(base_url)
    if u.scheme == "https":
        return http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def _chat(base_url, model, prompt, max_tokens, temperature, stream, stream_usage):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream_usage:
        body["stream_options"] = {"include_usage": True}

    conn = _connect(base_url)
    t0 = time.time()
    conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()

    if not stream:
        raw = resp.read().decode()
        conn.close()
        j = json.loads(raw)
        wall = time.time() - t0
        tokens = j["usage"]["completion_tokens"]
        return tokens, wall, None  # (tokens, wall, ttft)

    # Streaming: parse SSE, measure TTFT and decode-phase wall time.
    ttft = None
    first = None
    last = None
    completion_tokens = None

    def _delta_text(delta):
        # Qwen3.5 may split thinking vs. answer into content / reasoning_content.
        return delta.get("content") or delta.get("reasoning_content") or ""

    while True:
        line = resp.readline()
        if not line:
            break
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data in ("[DONE]", ""):
            continue
        try:
            chunk = json.loads(data)
        except (ValueError, json.JSONDecodeError):
            continue

        # usage arrives in the final chunk when stream_options include_usage.
        if chunk.get("usage"):
            completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)

        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if _delta_text(delta):
            now = time.time()
            if ttft is None:
                ttft = now - t0
                first = now
            last = now

    conn.close()
    wall = time.time() - t0
    return completion_tokens, wall, ttft


def bench_nonstream(args):
    print(f"== throughput (non-stream), {args.runs} run(s) ==")
    speeds = []
    for i in range(args.runs):
        tokens, wall, _ = _chat(args.base_url, args.model, args.prompt,
                                args.max_tokens, args.temperature, False, False)
        s = tokens / wall
        speeds.append(s)
        print(f"  run {i + 1}: {tokens:5d} tokens in {wall:6.2f}s  -> {s:6.1f} tok/s")
    print(f"  avg {sum(speeds) / len(speeds):.1f} tok/s\n")


def bench_stream(args):
    print(f"== latency+decode (streaming), {args.runs} run(s) ==")
    for i in range(args.runs):
        tokens, wall, ttft = _chat(args.base_url, args.model, args.prompt,
                                   args.max_tokens, args.temperature, True, True)
        decode_tokps = None
        if tokens:
            decode_tokps = tokens / wall
        ttft_ms = ttft * 1000 if ttft is not None else float("nan")
        print(f"  run {i + 1}: TTFT {ttft_ms:7.0f} ms | {tokens or 0:5d} tokens "
              f"in {wall:6.2f}s -> {decode_tokps or 0:6.1f} tok/s")
    print()


def main():
    p = argparse.ArgumentParser(
        description="Single-stream decode perf test for the Qwen3 vLLM server."
    )
    p.add_argument("--base-url", default=os.environ.get(
        "QWEN3_BENCH_URL", "http://127.0.0.1:8000"))
    p.add_argument("--model", default=os.environ.get("QWEN3_BENCH_MODEL", "qwen3.8"))
    p.add_argument("--prompt", default=(
        "Explain the concept of backpropagation in machine learning in detail, "
        "with a step by step example."))
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-stream", action="store_true",
                   help="only non-stream throughput, skip TTFT measurement")
    args = p.parse_args()

    print(f"endpoint: {args.base_url}  model: {args.model}  max_tokens: {args.max_tokens}")
    print()
    bench_nonstream(args)
    if not args.no_stream:
        bench_stream(args)


if __name__ == "__main__":
    main()