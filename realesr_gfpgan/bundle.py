"""Checkpoint bundling.

v1 bundle (kept for compatibility):
    {"version","architecture","config","params_esrgan","params_gfpgan"}

v2 bundle adds a `variants` map so one file can carry more than one ESRGAN
backbone (for example `RealESRGAN_x2plus` for 2x and `4x-UltraSharp` for 4x),
plus the architecture names spandrel needs to rebuild each one:

    {"version","architecture","config","variants":{"x2":state,"x4":state},
     "esrgan_arch":{"x2":"ESRGAN","x4":"ESRGAN"},"params_gfpgan"}
"""
import os

import torch


def _state(ckpt, preferred=("params_ema", "params")):
    if isinstance(ckpt, dict):
        for key in preferred:
            if key in ckpt:
                return ckpt[key]
    return ckpt


def create_unified_checkpoint(
    esrgan_path="experiments/pretrained_models/RealESRGAN_x2plus.pth",
    gfpgan_path="experiments/pretrained_models/GFPGANv1.4.pth",
    output_path="experiments/pretrained_models/RealESR_GFPGAN_v1.pth",
    config=None,
):
    """v1-format bundle: one ESRGAN backbone + GFPGAN weights."""
    print(f"Bundling RealESRGAN ({esrgan_path}) and GFPGAN ({gfpgan_path})...")

    state_esr = _state(torch.load(esrgan_path, map_location="cpu"))
    state_gfp = _state(torch.load(gfpgan_path, map_location="cpu"))

    bundle = {
        "version": "1.0.0",
        "architecture": "RealESR_GFPGAN_Unified",
        "config": config
        or {
            "upscale": 2,
            "fidelity_weight": 0.60,
            "micro_texture_alpha": 0.20,
            "blend_radius": 25,
            "tile": 400,
            "tile_pad": 16,
        },
        "params_esrgan": state_esr,
        "params_gfpgan": state_gfp,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(bundle, output_path)
    print(f"Unified checkpoint successfully created at: {output_path}")
    print(f"Total bundle size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def create_v2_checkpoint(
    backbones=None,
    gfpgan_path="experiments/pretrained_models/GFPGANv1.4.pth",
    output_path="experiments/pretrained_models/RealESR_GFPGAN_v2.pth",
    config=None,
):
    """v2 bundle: one or more ESRGAN backbones, selectable by name.

    ``backbones`` maps a variant name to ``(path, arch)``, e.g.
    ``{"x2": (".../RealESRGAN_x2plus.pth", "ESRGAN"),
       "x4": (".../4x-UltraSharp.pth", "ESRGAN")}``.
    """
    backbones = backbones or {
        "x2": ("experiments/pretrained_models/RealESRGAN_x2plus.pth", "ESRGAN"),
        "x4": ("experiments/pretrained_models/4x-UltraSharp.pth", "ESRGAN"),
    }

    variants, arch = {}, {}
    for name, (path, architecture) in backbones.items():
        print(f"Bundling {name}: {path}")
        variants[name] = _state(torch.load(path, map_location="cpu"))
        arch[name] = architecture

    state_gfp = _state(torch.load(gfpgan_path, map_location="cpu"))

    bundle = {
        "version": "2.0.0",
        "architecture": "RealESR_GFPGAN_v2",
        "config": config
        or {
            "default_variant": "x2",
            "fidelity_weight": 0.70,
            "micro_texture_alpha": 0.30,
            "face_mix": 0.80,
            "mask_threshold": 15,
            "tile": 384,
            "tile_pad": 32,
            "half": False,
        },
        "variants": variants,
        "esrgan_arch": arch,
        "params_gfpgan": state_gfp,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(bundle, output_path)
    print(f"v2 checkpoint created at: {output_path}")
    print(f"Total bundle size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    create_v2_checkpoint()
