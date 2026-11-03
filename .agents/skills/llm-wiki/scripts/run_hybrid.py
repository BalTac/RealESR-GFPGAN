import sys
import os
import time
import cv2
import numpy as np

# Torchvision monkeypatch for compatibility
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F

import torch
torch.backends.mkldnn.enabled = False

from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

def run_realesrgan_upscale(img, model, tile=400, pad=16, scale=2):
    """
    Tiled Real-ESRGAN inference on CPU with zero negative strides.
    """
    h, w = img.shape[:2]
    out_h, out_w = h * scale, w * scale
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    with torch.no_grad():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y_end = min(y + tile, h)
                x_end = min(x + tile, w)
                y_pad_start = max(0, y - pad)
                x_pad_start = max(0, x - pad)
                y_pad_end = min(h, y_end + pad)
                x_pad_end = min(w, x_end + pad)

                patch = img[y_pad_start:y_pad_end, x_pad_start:x_pad_end]
                # Convert BGR to RGB contiguously
                patch_rgb = np.ascontiguousarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                patch_t = torch.from_numpy(patch_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

                pred_t = model(patch_t)
                pred = (pred_t.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0)
                pred_bgr = np.ascontiguousarray(cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2BGR))

                # Crop padding
                crop_y_start = (y - y_pad_start) * scale
                crop_x_start = (x - x_pad_start) * scale
                crop_y_end = crop_y_start + (y_end - y) * scale
                crop_x_end = crop_x_start + (x_end - x) * scale

                out[y*scale:y_end*scale, x*scale:x_end*scale] = pred_bgr[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    return out

def main():
    device = torch.device('cpu')
    print(f"[Device] Using: {device}")

    # 1. Load RealESRGAN x2plus
    print("[Init] Loading RealESRGAN_x2plus RRDBNet...")
    t0 = time.time()
    esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    ckpt = torch.load('experiments/pretrained_models/RealESRGAN_x2plus.pth', map_location='cpu')
    key = 'params_ema' if 'params_ema' in ckpt else 'params'
    esrgan_model.load_state_dict(ckpt[key], strict=True)
    esrgan_model.eval()
    print(f"[Init] RealESRGAN loaded in {time.time() - t0:.2f}s")

    # 2. Load GFPGAN v1.4
    print("[Init] Loading GFPGANv1.4...")
    t1 = time.time()
    gfpgan_model_path = os.path.join('experiments', 'pretrained_models', 'GFPGANv1.4.pth')
    restorer = GFPGANer(
        model_path=gfpgan_model_path,
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device=device
    )
    print(f"[Init] GFPGAN loaded in {time.time() - t1:.2f}s")

    os.makedirs('results/hybrid', exist_ok=True)

    test_images = [
        ('Foto_3', 'inputs/whole_imgs/Foto_3.jpg'),
        ('Blake_Lively', 'inputs/whole_imgs/Blake_Lively.jpg'),
        ('10045', 'inputs/whole_imgs/10045.png'),
        ('00', 'inputs/whole_imgs/00.jpg')
    ]

    for name, img_path in test_images:
        ext = os.path.splitext(img_path)[1]
        print(f"\n==========================================")
        print(f"Running Hybrid Pipeline on {name} ({img_path})")
        print(f"==========================================")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error loading {img_path}")
            continue

        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale_in = 800.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale_in), int(h * scale_in)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            print(f"  Scaled input to {w}x{h} px for optimal CPU latency")
        else:
            print(f"  Input size: {w}x{h} px")

        start_total = time.time()

        # Step 1: Real-ESRGAN Background Enhancement
        t_bg = time.time()
        print("  1. Enhancing background with Real-ESRGAN x2plus...")
        bg_enhanced = run_realesrgan_upscale(img, esrgan_model, tile=400, pad=16, scale=2)
        elapsed_bg = time.time() - t_bg
        print(f"     -> Real-ESRGAN done in {elapsed_bg:.2f}s ({bg_enhanced.shape[1]}x{bg_enhanced.shape[0]})")

        # Step 2: GFPGAN Calibrated Face Pass
        t_face = time.time()
        print("  2. Restoring faces with Calibrated GFPGAN...")
        _, _, r_img_calib = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.60
        )
        # Micro-texture injection to avoid plastic effect
        gaussian = cv2.GaussianBlur(r_img_calib, (0, 0), sigmaX=1.2)
        unsharp_face = cv2.addWeighted(r_img_calib, 1.20, gaussian, -0.20, 0)
        elapsed_face = time.time() - t_face
        print(f"     -> GFPGAN Face done in {elapsed_face:.2f}s")

        # Step 3: Face Difference Mask & Seamless Blend onto RealESRGAN
        t_blend = time.time()
        h_t, w_t = bg_enhanced.shape[:2]
        if unsharp_face.shape[:2] != (h_t, w_t):
            unsharp_face = cv2.resize(unsharp_face, (w_t, h_t), interpolation=cv2.INTER_LANCZOS4)

        base_clean = cv2.resize(img, (w_t, h_t), interpolation=cv2.INTER_LANCZOS4)
        diff = cv2.absdiff(unsharp_face, base_clean)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, face_mask = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY)
        # Dilate and Gaussian blur for continuous natural falloff
        face_mask = cv2.dilate(face_mask, np.ones((15, 15), np.uint8), iterations=2)
        face_mask = cv2.GaussianBlur(face_mask, (25, 25), 0)
        mask_3d = face_mask[:, :, np.newaxis] / 255.0

        # Composite: Calibrated Face on RealESRGAN Background
        hybrid_img = (unsharp_face * mask_3d + bg_enhanced * (1.0 - mask_3d)).astype(np.uint8)
        elapsed_blend = time.time() - t_blend
        elapsed_total = time.time() - start_total

        out_path = f"results/hybrid/hybrid_{name}{ext}"
        cv2.imwrite(out_path, hybrid_img)
        print(f"  [SAVED] {out_path} in {elapsed_total:.2f}s (BG: {elapsed_bg:.1f}s, Face: {elapsed_face:.1f}s, Blend: {elapsed_blend:.2f}s)")

if __name__ == '__main__':
    main()
