"""v2 restoration pipeline: Real-ESRGAN (spandrel) + GFPGAN, no basicsr.

Differences from `model.py` (v1):

* the ESRGAN backbone is loaded with `spandrel`, so no `basicsr` import chain and
  no `torchvision.transforms.functional_tensor` monkeypatch;
* the backend model is pluggable: any ESRGAN/RRDB checkpoint spandrel understands
  (`RealESRGAN_x2plus`, `4x-UltraSharp`, ...) works unchanged;
* tiled inference blends with an overlap accumulator instead of hard seams;
* the face region is blended with the upscaled background by a configurable
  `face_mix`, which keeps some of the original grain instead of the smooth
  StyleGAN prior;
* fp16 is opt-in and off by default: on GPUs without tensor cores (Turing TU116
  and older) it is 4x *slower* than fp32.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import cv2
import numpy as np
import torch

__all__ = ["RestorePipeline", "load_bundle"]


def _to_bgr(x: torch.Tensor) -> np.ndarray:
    arr = (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round()
    return arr.astype(np.uint8)[:, :, ::-1].copy()


def load_bundle(path: str, device: torch.device, prefer: str = "x2") -> dict[str, Any]:
    """Load a unified checkpoint and pick the ESRGAN state dict to use.

    Returns ``{"state_esrgan": dict, "state_gfpgan": dict | None, "config": dict}``.
    v1 bundles only carry ``params_esrgan``/``params_gfpgan``; v2 bundles add
    ``esrgan_arch`` and a ``variants`` map with more than one backbone.
    """
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: not a unified checkpoint")

    variants = ckpt.get("variants") or {}
    if variants:
        name = prefer if prefer in variants else next(iter(variants))
        state_esrgan = variants[name]
        arch = (ckpt.get("esrgan_arch") or {}).get(name, "unknown")
    else:
        state_esrgan = ckpt.get("params_esrgan") or ckpt.get("params")
        arch = ckpt.get("esrgan_arch", "unknown")
        name = "default"
    if state_esrgan is None:
        raise ValueError(f"{path}: missing ESRGAN weights")

    return {
        "state_esrgan": state_esrgan,
        "state_gfpgan": ckpt.get("params_gfpgan"),
        "config": dict(ckpt.get("config") or {}),
        "variant": name,
        "arch": arch,
    }


class RestorePipeline:
    """Upscale + face restoration with boundary-aware fusion.

    Parameters
    ----------
    esrgan:
        Path to an ESRGAN/RRDB checkpoint understood by spandrel.
    gfpgan:
        Path to ``GFPGANv1.4.pth``.
    esrgan_state:
        Optional state dict to use instead of loading ``esrgan`` from disk (a
        bundle passed in memory). ``esrgan_arch`` must then be given.
    """

    def __init__(
        self,
        esrgan: Optional[str] = None,
        gfpgan: Optional[str] = None,
        device: str | torch.device | None = None,
        tile: int = 384,
        tile_pad: int = 32,
        half: bool = False,
        esrgan_state: Optional[dict] = None,
        esrgan_arch: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tile = int(tile)
        self.tile_pad = int(tile_pad)
        self.half = bool(half) and self.device.type == "cuda"

        if esrgan_state is not None:
            self.net, descale = _load_state_in_spandrel(esrgan_state, esrgan_arch, self.device)
            self.scale = descale
            self.model_path = f"<bundle:{esrgan_arch or 'unknown'}>"
        else:
            if not esrgan:
                raise ValueError("esrgan path or esrgan_state required")
            import spandrel

            desc = spandrel.ModelLoader().load_from_file(esrgan)
            self.net = desc.model.eval().to(self.device)
            self.scale = int(desc.scale or 4)
            self.model_path = esrgan

        self.restorer = None
        if gfpgan:
            from gfpgan import GFPGANer

            self.restorer = GFPGANer(
                model_path=gfpgan,
                upscale=self.scale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device,
            )

    # ---------------------------------------------------------------- upscale
    def upscale(self, img_bgr: np.ndarray) -> np.ndarray:
        """Tiled ESRGAN pass. Returns BGR uint8 at ``scale`` times the input."""
        rgb = img_bgr[:, :, ::-1].copy()
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        _, _, h, w = x.shape
        s, tile, pad = self.scale, self.tile, self.tile_pad
        autocast = (
            torch.autocast("cuda", dtype=torch.float16) if self.half else torch.autocast("cpu", enabled=False)
        )

        if h <= tile and w <= tile:
            with torch.no_grad(), autocast:
                y = self.net(x).float()
            return _to_bgr(y)

        out = torch.zeros(1, 3, h * s, w * s, device=self.device)
        acc = torch.zeros(1, 1, h * s, w * s, device=self.device)
        step = max(8, tile - 2 * pad)
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                if y1 - y0 < 8 or x1 - x0 < 8:
                    continue
                with torch.no_grad(), autocast:
                    patch = self.net(x[:, :, y0:y1, x0:x1]).float()
                oy, ox = y0 * s, x0 * s
                out[:, :, oy:oy + patch.shape[2], ox:ox + patch.shape[3]] += patch
                acc[:, :, oy:oy + patch.shape[2], ox:ox + patch.shape[3]] += 1
        out = out / acc.clamp(min=1)
        return _to_bgr(out)

    # ---------------------------------------------------------------- restore
    def restore(
        self,
        img_bgr: np.ndarray,
        fidelity: float = 0.7,
        alpha: float = 0.3,
        face_mix: float = 0.8,
        threshold: int = 15,
    ) -> tuple[np.ndarray, int]:
        """Full pipeline. Returns ``(bgr_uint8, faces_found)``."""
        bg = self.upscale(img_bgr)

        if self.restorer is None:
            return bg, 0

        _, _, face_bgr = self.restorer.enhance(
            img_bgr.copy(),
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=fidelity,
        )
        faces = len(getattr(getattr(self.restorer, "face_helper", None), "all_landmarks_5", []) or [])
        if face_bgr is None:
            return bg, 0

        face = face_bgr[:, :, ::-1].copy()
        if face.shape[:2] != bg.shape[:2]:
            face = cv2.resize(face, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        if alpha > 0:  # calibrazione micro-texture: riporta rilievo sulla pelle
            blur = cv2.GaussianBlur(face, (0, 0), sigmaX=1.2)
            face = cv2.addWeighted(face, 1.0 + alpha, blur, -alpha, 0)

        ref = cv2.resize(img_bgr, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        diff = cv2.absdiff(face, ref)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=2)
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        m3 = (mask[:, :, np.newaxis] / 255.0).astype(np.float32)

        f = face.astype(np.float32)
        b = bg.astype(np.float32)
        if face_mix < 1.0:  # dentro la maschera: GFPGAN + grana dell'upscaler
            f = face_mix * f + (1.0 - face_mix) * b
        out = (f * m3 + b * (1.0 - m3)).clip(0, 255).astype(np.uint8)
        return out, faces

    # ---------------------------------------------------------------- helpers
    def resize_to(self, img_bgr: np.ndarray, scale: int) -> np.ndarray:
        """Rescale the result so the output is ``scale`` times the input."""
        if scale == self.scale:
            return img_bgr
        h, w = img_bgr.shape[:2]
        f = scale / self.scale
        return cv2.resize(img_bgr, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_LANCZOS4)

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        return self.restore(img_bgr)[0]


def _load_state_in_spandrel(state: dict, arch: Optional[str], device: torch.device):
    """Rebuild a spandrel model from a raw state dict (bundle variant)."""
    import spandrel

    try:
        import spandrel_extra_arches  # noqa: F401  (optional: more architectures)
    except ImportError:
        pass

    desc = spandrel.ModelLoader().load_from_state_dict(state)
    model = desc.model.eval().to(device)
    return model, int(desc.scale or 2)
