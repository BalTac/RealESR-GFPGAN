#!/usr/bin/env python
"""Bench: file-based backbones vs unified bundle, 2x and 4x.

    python scripts/bench.py --input inputs/whole_imgs/sample.jpg --repeat 2
"""
import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realesr_gfpgan.pipeline import RestorePipeline, load_bundle  # noqa: E402

P = os.path.join("experiments", "pretrained_models")
X2, X4, GFP = (os.path.join(P, n) for n in ("RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth", "GFPGANv1.4.pth"))
BUNDLE = os.path.join(P, "RealESR_GFPGAN_v2.pth")


def timed(pipe, img, scale, kwargs, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.time()
        out, faces = pipe.restore(img, **kwargs)
        out = pipe.resize_to(out, scale)
        times.append(time.time() - t0)
    return min(times), faces, out.shape[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="inputs/whole_imgs/sample.jpg")
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()

    img = cv2.imread(a.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {a.input}")
    h, w = img.shape[:2]
    print(f"input {a.input}: {w}x{h} px")

    rows = []
    for scale, esrgan in ((2, X2), (4, X4)):
        if not os.path.exists(esrgan):
            print(f"skip scale {scale}: {esrgan} missing")
            continue
        pipe = RestorePipeline(esrgan=esrgan, gfpgan=GFP, tile=384)
        ms, faces, size = timed(pipe, img, scale, {"fidelity": 0.7, "alpha": 0.3, "face_mix": 0.8}, a.repeat)
        rows.append({"backend": os.path.basename(esrgan), "scale": scale, "faces": faces,
                     "size": list(size), "ms": int(ms * 1000)})

    if os.path.exists(BUNDLE):
        for name, scale in (("x2", 2), ("x4", 4)):
            info = load_bundle(BUNDLE, device="cpu", prefer=name)
            pipe = RestorePipeline(esrgan_state=info["state_esrgan"], esrgan_arch=info["arch"],
                                   gfpgan=GFP, tile=384)
            ms, faces, size = timed(pipe, img, scale, {"fidelity": 0.7, "alpha": 0.3, "face_mix": 0.8}, a.repeat)
            rows.append({"backend": f"bundle:{name}", "scale": scale, "faces": faces,
                         "size": list(size), "ms": int(ms * 1000)})

    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
