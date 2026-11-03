import sys
import os
import time
import cv2
import numpy as np

# Torchvision compatibility monkeypatch
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F

import torch
import torch.nn as nn
torch.backends.mkldnn.enabled = False

from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

class RealESR_GFPGAN(nn.Module):
    """
    RealESR-GFPGAN Unified Image Restoration Model.
    
    Seamlessly combines:
      1. Real-ESRGAN (RRDBNet) for whole-image & background structural super-resolution.
      2. Calibrated GFPGAN v1.4 with high-frequency micro-texture injection for natural skin relief.
      3. Boundary-aware soft alpha blending to eliminate deconvolution and checkerboard seams.
    """
    def __init__(self, checkpoint_path=None, upscale=2, tile=400, tile_pad=16, device=None):
        super().__init__()
        self.upscale = upscale
        self.tile = tile
        self.tile_pad = tile_pad
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize Real-ESRGAN backbone
        self.esrgan = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=upscale
        ).to(self.device)
        self.esrgan.eval()

        # 2. Checkpoint loading
        self.restorer = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_unified_checkpoint(checkpoint_path)

    def load_unified_checkpoint(self, path):
        """
        Loads parameters from a single unified checkpoint or standard pth.
        """
        print(f"[RealESR-GFPGAN] Loading unified checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        
        # Check if bundle format or standard
        if 'params_esrgan' in ckpt:
            self.esrgan.load_state_dict(ckpt['params_esrgan'], strict=True)
            gfpgan_weights = ckpt.get('gfpgan_temp_path', os.path.join('experiments', 'pretrained_models', 'GFPGANv1.4.pth'))
        else:
            # Fallback for standalone RealESRGAN ckpt
            key = 'params_ema' if 'params_ema' in ckpt else 'params'
            self.esrgan.load_state_dict(ckpt[key] if key in ckpt else ckpt, strict=False)
            gfpgan_weights = os.path.join('experiments', 'pretrained_models', 'GFPGANv1.4.pth')

        # Initialize GFPGAN face restorer
        self.restorer = GFPGANer(
            model_path=gfpgan_weights,
            upscale=self.upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=self.device
        )
        print("[RealESR-GFPGAN] Unified pipeline loaded successfully.")

    def _upscale_background(self, img):
        """Tiled Real-ESRGAN inference on CPU/GPU."""
        h, w = img.shape[:2]
        out_h, out_w = h * self.upscale, w * self.upscale
        out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        with torch.no_grad():
            for y in range(0, h, self.tile):
                for x in range(0, w, self.tile):
                    y_end = min(y + self.tile, h)
                    x_end = min(x + self.tile, w)
                    y_pad_start = max(0, y - self.tile_pad)
                    x_pad_start = max(0, x - self.tile_pad)
                    y_pad_end = min(h, y_end + self.tile_pad)
                    x_pad_end = min(w, x_end + self.tile_pad)

                    patch = img[y_pad_start:y_pad_end, x_pad_start:x_pad_end]
                    patch_rgb = np.ascontiguousarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                    patch_t = torch.from_numpy(patch_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0

                    pred_t = self.esrgan(patch_t)
                    pred = (pred_t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0)
                    pred_bgr = np.ascontiguousarray(cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2BGR))

                    crop_y_start = (y - y_pad_start) * self.upscale
                    crop_x_start = (x - x_pad_start) * self.upscale
                    crop_y_end = crop_y_start + (y_end - y) * self.upscale
                    crop_x_end = crop_x_start + (x_end - x) * self.upscale

                    out[y*self.upscale:y_end*self.upscale, x*self.upscale:x_end*self.upscale] = pred_bgr[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        return out

    def enhance(self, img, fidelity_weight=0.60, texture_injection=True):
        """
        Executes unified restoration on an input image.
        Returns:
            restored_img (np.ndarray): High-resolution restored image (BGR uint8)
        """
        # Step 1: Real-ESRGAN Background Enhancement
        bg_enhanced = self._upscale_background(img)

        # Step 2: Calibrated GFPGAN Face Restoration
        _, _, r_img_calib = self.restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=fidelity_weight
        )

        if texture_injection:
            # Subtle unsharp mask to restore organic skin micro-relief
            gaussian = cv2.GaussianBlur(r_img_calib, (0, 0), sigmaX=1.2)
            unsharp_face = cv2.addWeighted(r_img_calib, 1.20, gaussian, -0.20, 0)
        else:
            unsharp_face = r_img_calib

        # Step 3: Seamless Boundary-Aware Composite
        h_t, w_t = bg_enhanced.shape[:2]
        if unsharp_face.shape[:2] != (h_t, w_t):
            unsharp_face = cv2.resize(unsharp_face, (w_t, h_t), interpolation=cv2.INTER_LANCZOS4)

        base_clean = cv2.resize(img, (w_t, h_t), interpolation=cv2.INTER_LANCZOS4)
        diff = cv2.absdiff(unsharp_face, base_clean)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, face_mask = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY)
        face_mask = cv2.dilate(face_mask, np.ones((15, 15), np.uint8), iterations=2)
        face_mask = cv2.GaussianBlur(face_mask, (25, 25), 0)
        mask_3d = face_mask[:, :, np.newaxis] / 255.0

        hybrid_output = (unsharp_face * mask_3d + bg_enhanced * (1.0 - mask_3d)).astype(np.uint8)
        return hybrid_output

    def __call__(self, img):
        return self.enhance(img)
