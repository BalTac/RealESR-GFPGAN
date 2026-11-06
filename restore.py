#!/usr/bin/env python
"""restore.py — v2 CLI for RealESR-GFPGAN (spandrel backend, no basicsr).

    python restore.py -i in.jpg -o out.jpg                # 2x, x2plus
    python restore.py -i in.jpg -o out.jpg --scale 4      # 4x, 4x-UltraSharp
    python restore.py -i in.jpg -o out.jpg --fidelity .7 --alpha .3 --face-mix .8
    python restore.py -i in.jpg -o out.jpg --bundle experiments/pretrained_models/RealESR_GFPGAN_v2.pth

Equivalent to the original `inference.py`, but the ESRGAN backbone is pluggable
and the face/background fusion is configurable.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2

from realesr_gfpgan.pipeline import RestorePipeline, load_bundle

DEFAULT_X2 = "experiments/pretrained_models/RealESRGAN_x2plus.pth"
DEFAULT_X4 = "experiments/pretrained_models/4x-UltraSharp.pth"
DEFAULT_GFPGAN = "experiments/pretrained_models/GFPGANv1.4.pth"


def main() -> None:
    p = argparse.ArgumentParser(description="RealESR-GFPGAN v2 restoration")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", default="results/output.jpg")
    p.add_argument("-s", "--scale", type=int, default=2, choices=(2, 4))
    p.add_argument("-w", "--fidelity", type=float, default=0.7, help="GFPGAN fidelity weight (higher = closer to input)")
    p.add_argument("--alpha", type=float, default=0.3, help="micro-texture injection (0 disables)")
    p.add_argument("--face-mix", type=float, default=0.8, dest="face_mix",
                   help="inside the mask: 1.0 = GFPGAN only, 0.8 = + upscaler grain")
    p.add_argument("--threshold", type=int, default=15)
    p.add_argument("--tile", type=int, default=384)
    p.add_argument("--esrgan", help="ESRGAN/RRDB checkpoint (overrides --scale default)")
    p.add_argument("--gfpgan", default=DEFAULT_GFPGAN)
    p.add_argument("--bundle", help="unified checkpoint (.pth) instead of --esrgan")
    p.add_argument("--half", action="store_true", help="fp16 autocast (slower on GPUs without tensor cores)")
    p.add_argument("--device", default=None)
    p.add_argument("--json", action="store_true", help="print a JSON line instead of text")
    a = p.parse_args()

    if not os.path.exists(a.input):
        raise SystemExit(f"input not found: {a.input}")

    t0 = time.time()
    if a.bundle:
        info = load_bundle(a.bundle, device=a.device or "cpu", prefer=f"x{a.scale}")
        pipe = RestorePipeline(
            esrgan_state=info["state_esrgan"],
            esrgan_arch=info["arch"],
            gfpgan=a.gfpgan,
            device=a.device,
            tile=a.tile,
            half=a.half,
        )
    else:
        esrgan = a.esrgan or (DEFAULT_X2 if a.scale == 2 else DEFAULT_X4)
        pipe = RestorePipeline(esrgan=esrgan, gfpgan=a.gfpgan, device=a.device, tile=a.tile, half=a.half)

    img = cv2.imread(a.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image: {a.input}")

    out, faces = pipe.restore(
        img, fidelity=a.fidelity, alpha=a.alpha, face_mix=a.face_mix, threshold=a.threshold
    )
    out = pipe.resize_to(out, a.scale)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    cv2.imwrite(a.output, out)

    elapsed = time.time() - t0
    info = {
        "ok": True,
        "input": a.input,
        "output": a.output,
        "size": [int(out.shape[1]), int(out.shape[0])],
        "faces": faces,
        "scale": a.scale,
        "fidelity": a.fidelity,
        "alpha": a.alpha,
        "face_mix": a.face_mix,
        "backend": pipe.model_path,
        "device": str(pipe.device),
        "ms": int(elapsed * 1000),
    }
    print(json.dumps(info, ensure_ascii=False) if a.json else _pretty(info))


def _pretty(i: dict) -> str:
    return (
        f"{i['input']} -> {i['output']}  {i['size'][0]}x{i['size'][1]} px\n"
        f"  faces {i['faces']} · scale {i['scale']} · fidelity {i['fidelity']} · "
        f"alpha {i['alpha']} · face-mix {i['face_mix']}\n"
        f"  backend {i['backend']} on {i['device']} · {i['ms'] / 1000:.1f}s"
    )


if __name__ == "__main__":
    main()
