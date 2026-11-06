#!/usr/bin/env python
"""Self-check for the v2 pipeline. No pytest, no GPU required for the assertions.

    python scripts/selfcheck.py

Fails loudly if the tiled upscaler changes geometry, if `resize_to` is off, or if
the v2 bundle cannot be rebuilt through spandrel.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realesr_gfpgan.pipeline import RestorePipeline, load_bundle  # noqa: E402

X2 = os.path.join("experiments", "pretrained_models", "RealESRGAN_x2plus.pth")
BUNDLE = os.path.join("experiments", "pretrained_models", "RealESR_GFPGAN_v2.pth")


def check_upscale_geometry():
    pipe = RestorePipeline(esrgan=X2, gfpgan=None, device="cpu", tile=32, tile_pad=8)
    img = np.random.default_rng(0).integers(0, 255, (64, 96, 3), dtype=np.uint8)
    out = pipe.upscale(img)
    assert out.shape[:2] == (64 * pipe.scale, 96 * pipe.scale), out.shape
    assert out.dtype == np.uint8
    print(f"ok  upscale geometry {img.shape[:2]} -> {out.shape[:2]} (scale {pipe.scale})")


def check_tiling_matches_single_pass():
    """Tiled output must be close to a single-pass output (same weights)."""
    # gradiente + blocco: caso realistico; il rumore puro e' il caso peggiore
    yy, xx = np.mgrid[0:96, 0:96]
    img = np.stack([(xx * 2) % 256, (yy * 2) % 256, ((xx + yy) * 1.3) % 256], axis=-1).astype(np.uint8)
    img[20:60, 20:70] = (200, 40, 40)
    big = RestorePipeline(esrgan=X2, gfpgan=None, device="cpu", tile=4096, tile_pad=8)
    small = RestorePipeline(esrgan=X2, gfpgan=None, device="cpu", tile=48, tile_pad=16)
    a = big.upscale(img).astype(np.int16)
    b = small.upscale(img).astype(np.int16)
    diff = float(np.abs(a - b).mean())
    assert diff < 3.0, f"tiling diverges: mean |diff| = {diff:.2f}"
    print(f"ok  tiling vs single pass: mean |diff| = {diff:.2f}/255")


def check_resize_to():
    pipe = RestorePipeline(esrgan=X2, gfpgan=None, device="cpu", tile=4096)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = pipe.upscale(img)
    two = pipe.resize_to(out, 2)
    four = pipe.resize_to(out, 4)
    assert two.shape[:2] == (128, 128), two.shape
    assert four.shape[:2] == (256, 256), four.shape
    print("ok  resize_to 2x/4x")


def check_bundle_roundtrip():
    if not os.path.exists(BUNDLE):
        print(f"skip bundle: {BUNDLE} not found (build it with bundle.py)")
        return
    info = load_bundle(BUNDLE, device=torch.device("cpu"), prefer="x2")
    assert info["state_esrgan"] and info["state_gfpgan"], "bundle missing weights"
    pipe = RestorePipeline(
        esrgan_state=info["state_esrgan"], esrgan_arch=info["arch"], gfpgan=None, device="cpu", tile=4096
    )
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    out = pipe.upscale(img)
    assert out.shape[0] == 32 * pipe.scale, out.shape
    print(f"ok  bundle roundtrip: variant {info['variant']} arch {info['arch']} scale {pipe.scale}")


if __name__ == "__main__":
    check_upscale_geometry()
    check_tiling_matches_single_pass()
    check_resize_to()
    check_bundle_roundtrip()
    print("all checks passed")
