#!/usr/bin/env python
"""No-reference evaluation (NIQE / BRISQUE / MUSIQ) for the v1 and v2 fusion.

    python scripts/eval_metrics.py --out results/metrics.json

Every output is resized so its longest side is `--size` (default 1024) before the
metrics are computed: NIQE and BRISQUE are resolution dependent, and comparing a
2x output against a 4x output without normalising the size would measure the scale
rather than the restoration.

Variants scored:
  input        original file, untouched
  gfpgan       bicubic 2x + GFPGAN (fidelity 0.7)  -> "face prior alone"
  esrgan       plain backbone upscale, no face pass -> "scene prior alone"
  v1_fusion    fidelity 0.6, alpha 0.2, face_mix 1.0
  v2_fusion    fidelity 0.7, alpha 0.3, face_mix 0.8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realesr_gfpgan.pipeline import RestorePipeline  # noqa: E402

P = os.path.join("experiments", "pretrained_models")
X2 = os.path.join(P, "RealESRGAN_x2plus.pth")
GFP = os.path.join(P, "GFPGANv1.4.pth")


def normalise(img, size):
    h, w = img.shape[:2]
    f = size / max(h, w)
    if f >= 1:
        return img
    return cv2.resize(img, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)


def gfpgan_only(img, size):
    from gfpgan import GFPGANer

    up = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    r = GFPGANer(model_path=GFP, upscale=2, arch="clean", channel_multiplier=2,
                 bg_upsampler=None, device="cuda" if torch.cuda.is_available() else "cpu")
    _, _, out = r.enhance(up, has_aligned=False, only_center_face=False, paste_back=True, weight=0.7)
    return out if out is not None else up


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=[
        "inputs/whole_imgs/Blake_Lively.jpg",
        "inputs/whole_imgs/10045.png",
        "inputs/whole_imgs/00.jpg",
    ])
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", default="results/metrics.json")
    ap.add_argument("--save-dir", default="results/metrics")
    a = ap.parse_args()

    import pyiqa

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = {n: pyiqa.create_metric(n, device=dev) for n in ("niqe", "brisque", "musiq")}

    pipe = RestorePipeline(esrgan=X2, gfpgan=GFP, tile=384)
    os.makedirs(a.save_dir, exist_ok=True)

    rows = []
    for path in a.inputs:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip {path}", file=sys.stderr)
            continue
        name = os.path.splitext(os.path.basename(path))[0]

        v1, _ = pipe.restore(img, fidelity=0.6, alpha=0.2, face_mix=1.0)
        v2, _ = pipe.restore(img, fidelity=0.7, alpha=0.3, face_mix=0.8)
        outputs = {
            "input": img,
            "gfpgan": gfpgan_only(img, a.size),
            "esrgan": pipe.upscale(img),
            "v1_fusion": v1,
            "v2_fusion": v2,
        }

        for variant, out in outputs.items():
            out = pipe.resize_to(out, a.scale) if variant not in ("input", "gfpgan") else out
            cv2.imwrite(os.path.join(a.save_dir, f"{name}_{variant}.png"), out)
            norm = normalise(out, a.size)
            cv2.imwrite(os.path.join(a.save_dir, f"{name}_{variant}_norm.png"), norm)
            tensor = torch.from_numpy(norm[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
            scores = {k: float(m(tensor).item()) for k, m in metrics.items()}
            rows.append({"image": name, "variant": variant, "size": [int(out.shape[1]), int(out.shape[0])], **scores})
            print(json.dumps(rows[-1]))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
