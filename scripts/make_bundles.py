#!/usr/bin/env python
"""Build the two release bundles.

    python scripts/make_bundles.py

* RealESR_GFPGAN_v2.pth             x2 = RealESRGAN_x2plus, x4 = RealESRGAN_x4plus
                                    weights: BSD-3 (xinntao) + Apache-2.0 (TencentARC)
* RealESR_GFPGAN_v2_ultrasharp.pth  x4 = 4x-UltraSharp
                                    weights: CC BY-NC-SA 4.0 (Kim2091) -> NON commerciale

UltraSharp is not fetched automatically: download it from Hugging Face
(Kim2091/UltraSharp) after reading its licence and drop it in
experiments/pretrained_models/4x-UltraSharp.pth.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realesr_gfpgan.bundle import create_v2_checkpoint

P = os.path.join("experiments", "pretrained_models")
GFP = os.path.join(P, "GFPGANv1.4.pth")


def main():
    permissive = {
        "x2": (os.path.join(P, "RealESRGAN_x2plus.pth"), "ESRGAN"),
        "x4": (os.path.join(P, "RealESRGAN_x4plus.pth"), "ESRGAN"),
    }
    missing = [p for p, _ in permissive.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    create_v2_checkpoint(backbones=permissive, gfpgan_path=GFP,
                         output_path=os.path.join(P, "RealESR_GFPGAN_v2.pth"))

    ultra = os.path.join(P, "4x-UltraSharp.pth")
    if not os.path.exists(ultra):
        print("\nskip NC bundle: 4x-UltraSharp.pth not present (see the licence first)")
        return
    create_v2_checkpoint(backbones={"x4": (ultra, "ESRGAN")}, gfpgan_path=GFP,
                         output_path=os.path.join(P, "RealESR_GFPGAN_v2_ultrasharp.pth"),
                         config={"default_variant": "x4", "fidelity_weight": 0.70,
                                 "micro_texture_alpha": 0.30, "face_mix": 0.80,
                                 "mask_threshold": 15, "tile": 384, "tile_pad": 32,
                                 "half": False, "license": "CC-BY-NC-SA-4.0"})


if __name__ == "__main__":
    main()
