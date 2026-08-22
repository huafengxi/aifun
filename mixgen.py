#!/home/yuanqi.xhf/miniconda3/bin/python
"""
mixgen.py — reference-image-conditioned generation (manual img2img) on the
local Krea 2 diffusers pipeline.

Krea-2 is a T2I-only model (no native img2img in diffusers/SGLang), so this
script implements the community-standard manual img2img trick (same idea as
InvokeAI/ComfyUI): VAE-encode the reference image, normalize with the VAE's
latents_mean/std, pack latents like Krea2Pipeline._pack_latents, mix with
noise at sigma_0 = truncated_schedule[0], then run the pipeline with the
truncated sigma schedule via its public `latents`/`sigmas` args.

No changes to paint.py are needed; the pipeline context is borrowed from
paint.load_krea2_pipeline (dual-GPU aware).

Usage:
    ./mixgen.py --ref ref.jpg --strength 0.55 --seed 42 \
        --width 1024 --height 1792 -o out.jpg "a prompt"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint  # noqa: E402


def encode_ref(ctx, ref_path, width, height):
    """VAE-encode a reference image (resized to width x height) into packed
    Krea-2 latents, normalized the same way the pipeline denormalizes on
    decode (latents*std + mean on decode => (z-mean)/std on encode)."""
    import numpy as np
    import torch
    from PIL import Image

    pipe = ctx["pipe"]
    vae_dev = next(pipe.vae.parameters()).device
    img = Image.open(ref_path).convert("RGB").resize((width, height),
                                                     Image.LANCZOS)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
    x = x.permute(2, 0, 1)[None, :, None, :, :]  # (1, C, 1, H, W) temporal dim
    x = x.to(device=vae_dev, dtype=pipe.vae.dtype)
    with torch.no_grad():
        z = pipe.vae.encode(x).latent_dist.mode()
    mean = torch.tensor(pipe.vae.config.latents_mean,
                        device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(pipe.vae.config.latents_std,
                       device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    z = (z - mean) / std
    z = z[:, :, 0]  # drop temporal dim: (B, C, H, W)
    B, C, H, W = z.shape
    return pipe._pack_latents(z, B, C, H, W)


def render_i2i(ctx, prompt, ref_path, width, height, strength, seed,
               steps=8):
    import numpy as np
    import torch

    pipe = ctx["pipe"]
    dev0 = torch.device(ctx["device"])
    packed = encode_ref(ctx, ref_path, width, height).to(dev0)
    gen = torch.Generator(device=dev0).manual_seed(seed)
    noise = torch.randn(packed.shape, generator=gen, device=dev0,
                        dtype=packed.dtype)
    full = list(np.linspace(1.0, 1.0 / steps, steps))
    k = max(1, min(steps, int(round(steps * strength))))
    sigmas = full[-k:]
    s0 = sigmas[0]
    latents = (1.0 - s0) * packed + s0 * noise

    kwargs = dict(width=width, height=height, num_inference_steps=steps,
                  guidance_scale=0.0, sigmas=sigmas, latents=latents)
    call_args = (prompt,)
    if ctx["dual"]:
        # text encoder lives on the other GPU (see paint.load_krea2_pipeline)
        emb_dev = torch.device(ctx["dev1"])
        pe, pem = pipe.encode_prompt(prompt=prompt, device=emb_dev,
                                     num_images_per_prompt=1)
        kwargs["prompt_embeds"] = pe.to(dev0)
        kwargs["prompt_embeds_mask"] = pem.to(dev0)
        call_args = ()
    out = pipe(*call_args, **kwargs, **ctx["extra"]).images
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--spec", help="JSON file: list of {name, ref, prompt, "
                                    "seed, strength[, width, height, steps]}; "
                                    "loads the model once and renders all")
    ap.add_argument("--ref")
    ap.add_argument("-o", "--output")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1792)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--dual-gpu", action="store_true")
    ap.add_argument("--model", default="krea2")
    args = ap.parse_args()

    model = paint.ALL_ALIASES.get(args.model, args.model)
    if model in paint.KREA2_LORAS:
        sys.exit("LoRA aliases not supported by mixgen; use the base model")

    if not args.spec and not (args.prompt and args.ref and args.output):
        ap.error("single-shot mode needs PROMPT, --ref and -o")

    if args.spec:
        import json
        jobs = json.load(open(args.spec))
        ctx = paint.load_krea2_pipeline(model, dual_gpu=args.dual_gpu)
        for j in jobs:
            imgs = render_i2i(ctx, j["prompt"], os.path.expanduser(j["ref"]),
                              j.get("width", args.width),
                              j.get("height", args.height),
                              j.get("strength", args.strength),
                              j["seed"], j.get("steps", args.steps))
            out = os.path.expanduser(j["out"])
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            imgs[0].save(out, quality=args.quality)
            print(f"saved {out}", flush=True)
        return

    ctx = paint.load_krea2_pipeline(model, dual_gpu=args.dual_gpu)
    imgs = render_i2i(ctx, args.prompt, args.ref, args.width, args.height,
                      args.strength, args.seed, args.steps)
    out = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imgs[0].save(out, quality=args.quality)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
