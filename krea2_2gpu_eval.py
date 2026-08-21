#!/usr/bin/env python3
"""krea2-2gpu-eval (task 2026-08-21-17-12-57-24ge)

Evaluate whether a 2nd GPU helps Krea-2-Turbo FP8 inference.

Modes:
  single   baseline: whole pipeline on one GPU (same path as paint.py)
  devmap   component-level device_map="balanced" across both GPUs
  manual   manual component split: transformer@GPU0, text_encoder+vae@GPU1

Prints load time / gen time / per-GPU peak memory as JSON on the last line.
"""
import argparse
import gc
import json
import os
import sys
import threading
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paint import resolve_model_path, _load_krea2_fp8_transformer, FP8_BASE  # noqa: E402

FP8_ID = "sakamakismile/Krea-2-Turbo-FP8"
PROMPT = "a cat sitting on a windowsill, soft morning light"
SEED = 42


def gpu_smi_sampler(interval=0.5, stop=None):
    """Background sampler of nvidia-smi used memory per GPU."""
    import subprocess
    peaks = {}
    while not stop.is_set():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout
        for line in out.strip().splitlines():
            idx, used = line.split(",")
            peaks[int(idx)] = max(peaks.get(int(idx), 0), int(used))
        stop.wait(interval)
    return peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "devmap", "manual"], required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default="/tmp/krea2_2gpu_eval.png")
    a = ap.parse_args()

    import torch
    from diffusers import Krea2Pipeline

    n_gpu = torch.cuda.device_count()
    smi_base = {}
    import subprocess
    for line in subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines():
        idx, used = line.split(",")
        smi_base[int(idx)] = int(used)

    stop = threading.Event()
    smi_peaks = {}
    t = threading.Thread(
        target=lambda: smi_peaks.update(gpu_smi_sampler(stop=stop)), daemon=True)
    t.start()

    fp8_path = resolve_model_path(FP8_ID)
    base_path = resolve_model_path(FP8_BASE[FP8_ID])

    for i in range(n_gpu):
        torch.cuda.reset_peak_memory_stats(i)

    t0 = time.time()
    transformer = _load_krea2_fp8_transformer(base_path, fp8_path)
    if a.mode == "single":
        pipe = Krea2Pipeline.from_pretrained(
            base_path, transformer=transformer, torch_dtype=torch.bfloat16)
        # pick the GPU with most free memory (same as paint.py)
        best = max(range(n_gpu), key=lambda i: torch.cuda.mem_get_info(i)[0])
        device = f"cuda:{best}"
        print(f"single-GPU mode: .to({device})", file=sys.stderr)
        pipe.to(device)
    elif a.mode == "devmap":
        print("devmap mode: from_pretrained(device_map='balanced')", file=sys.stderr)
        pipe = Krea2Pipeline.from_pretrained(
            base_path, transformer=transformer, torch_dtype=torch.bfloat16,
            device_map="balanced")
        try:
            print("final device map:", pipe.hf_device_map, file=sys.stderr)
        except AttributeError:
            pass
        # NOTE: diffusers leaves a passed-in transformer on CPU; move per map
        dev = pipe.hf_device_map.get("transformer", 0) if hasattr(pipe, "hf_device_map") else 0
        pipe.transformer.to(f"cuda:{dev}")
        device = f"cuda:{dev}"
    else:
        pipe = Krea2Pipeline.from_pretrained(
            base_path, transformer=transformer, torch_dtype=torch.bfloat16)
        pipe.transformer.to("cuda:0")
        pipe.text_encoder.to("cuda:1")
        pipe.vae.to("cuda:1")
        device = "cuda:0"
    load_s = time.time() - t0
    print(f"LOAD_SECONDS={load_s:.1f}", file=sys.stderr)

    for i in range(n_gpu):
        torch.cuda.reset_peak_memory_stats(i)

    generator = torch.Generator("cpu").manual_seed(SEED)
    call_kwargs = dict(
        width=a.width, height=a.height,
        num_inference_steps=a.steps, guidance_scale=0.0,
        generator=generator,
    )
    if a.mode in ("manual", "devmap"):
        # VAE lives on the other GPU → move latents in before decode
        _orig_decode = pipe.vae.decode
        _vae_dev = next(pipe.vae.parameters()).device

        def _decode(z, **kw):
            return _orig_decode(z.to(_vae_dev), **kw)

        pipe.vae.decode = _decode
        # pipeline picks _execution_device from the first component (vae on
        # GPU1) → force denoise-side device for latents/position_ids/timesteps
        type(pipe)._execution_device = property(lambda self: torch.device(device))
        # encode on the text encoder's device, then move embeds to the
        # transformer's device (pipeline does no cross-device moves itself)
        enc_dev = str(next(pipe.text_encoder.parameters()).device)
        enc = pipe.encode_prompt(prompt=PROMPT, device=enc_dev, num_images_per_prompt=1)
        call_kwargs["prompt_embeds"] = enc[0].to(device)
        call_kwargs["prompt_embeds_mask"] = enc[1].to(device)
        call_args = ()
    else:
        call_args = (PROMPT,)
    t0 = time.time()
    err = None
    try:
        imgs = pipe(*call_args, **call_kwargs).images
        imgs[0].save(a.out)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        err = f"{type(e).__name__}: {str(e)[:300]}"
    gen_s = time.time() - t0

    stop.set(); t.join(timeout=2)
    torch_peak = {i: torch.cuda.max_memory_allocated(i) / 2**30 for i in range(n_gpu)}
    smi_delta = {i: (smi_peaks.get(i, smi_base[i]) - smi_base[i]) / 1024 for i in range(n_gpu)}
    gc.collect(); torch.cuda.empty_cache()

    result = {
        "mode": a.mode, "size": f"{a.width}x{a.height}", "steps": a.steps,
        "prompt": PROMPT, "seed": SEED,
        "load_s": round(load_s, 1), "gen_s": round(gen_s, 1),
        "error": err,
        "torch_peak_GiB": {k: round(v, 2) for k, v in torch_peak.items()},
        "smi_delta_GiB": {k: round(v, 2) for k, v in smi_delta.items()},
        "out": a.out,
    }
    print("RESULT_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
