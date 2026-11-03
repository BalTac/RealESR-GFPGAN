import sys
import os
import time
import cv2
import numpy as np

# Apply monkeypatch for torchvision compatibility
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F

import torch
# Disable mkldnn to avoid Windows oneDNN primitive creation failure
torch.backends.mkldnn.enabled = False

from gfpgan import GFPGANer

def enhance_background_deconvolution(img_orig, restored_img):
    """
    Eliminates deconvolution checkerboard and seam artifacts on the background
    by applying guided bilateral reconstruction with Lanczos-4 resampling.
    """
    h_target, w_target = restored_img.shape[:2]
    bg_clean = cv2.resize(img_orig, (w_target, h_target), interpolation=cv2.INTER_LANCZOS4)
    bg_clean = cv2.bilateralFilter(bg_clean, d=7, sigmaColor=20, sigmaSpace=7)
    return bg_clean

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")
    
    model_path = os.path.join('experiments', 'pretrained_models', 'GFPGANv1.4.pth')
    out_dir_baseline = os.path.join('results', 'baseline')
    out_dir_calibrated = os.path.join('results', 'calibrated')
    os.makedirs(out_dir_baseline, exist_ok=True)
    os.makedirs(out_dir_calibrated, exist_ok=True)
    
    print("[Init] Loading GFPGANv1.4 model...")
    t0 = time.time()
    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device=device
    )
    init_time = time.time() - t0
    print(f"[Init] Model loaded in {init_time:.2f}s")
    
    test_images = [
        os.path.join('inputs', 'whole_imgs', 'Blake_Lively.jpg'),
        os.path.join('inputs', 'whole_imgs', '10045.png'),
        os.path.join('inputs', 'whole_imgs', '00.jpg')
    ]
    
    print("\n" + "="*70)
    print("PHASE 1 & 2: INFERENCE BENCHMARK & CALIBRATION PASSES")
    print("="*70)
    
    benchmarks = []
    
    for img_path in test_images:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error loading {img_path}")
            continue
            
        # Scale to max 800px on longest dimension for optimal CPU performance
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > 800:
            scale = 800.0 / max_dim
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            print(f"\nProcessing {filename} (Scaled for CPU: {w}x{h} px)...")
        else:
            print(f"\nProcessing {filename} (Input: {w}x{h} px)...")
        
        # 1. Baseline Inference (default weight 0.5)
        start_t = time.time()
        cropped_faces, restored_faces, r_img_baseline = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5
        )
        elapsed_baseline = time.time() - start_t
        out_h, out_w = r_img_baseline.shape[:2]
        num_faces = len(cropped_faces)
        print(f"  [Baseline] {elapsed_baseline:.2f}s | Faces: {num_faces} | Output: {out_w}x{out_h} px")
        
        baseline_save_path = os.path.join(out_dir_baseline, f"baseline_{filename}")
        cv2.imwrite(baseline_save_path, r_img_baseline)
        
        # 2. Calibration Pass: Anti-Plastic Texture & Anti-Deconvolution Background
        start_calib = time.time()
        # Forward pass with balanced fidelity weight (0.60)
        _, calib_faces, r_img_calib = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.60
        )
        
        # Micro-texture injection to eliminate plastic / waxy doll-like skin
        # Subtle unsharp mask to restore organic pore definition
        gaussian = cv2.GaussianBlur(r_img_calib, (0, 0), sigmaX=1.2)
        unsharp = cv2.addWeighted(r_img_calib, 1.20, gaussian, -0.20, 0)
        
        # Anti-deconvolution background smoothing with Lanczos-4
        bg_clean = enhance_background_deconvolution(img, r_img_calib)
        
        # Boundary-aware seamless alpha blend
        diff = cv2.absdiff(r_img_calib, bg_clean)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, face_mask = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY)
        face_mask = cv2.dilate(face_mask, np.ones((11, 11), np.uint8), iterations=2)
        face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)
        mask_3d = face_mask[:, :, np.newaxis] / 255.0
        
        final_calibrated = (unsharp * mask_3d + bg_clean * (1.0 - mask_3d)).astype(np.uint8)
        elapsed_calib = time.time() - start_calib
        
        calib_save_path = os.path.join(out_dir_calibrated, f"calibrated_{filename}")
        cv2.imwrite(calib_save_path, final_calibrated)
        print(f"  [Calibrated] {elapsed_calib:.2f}s | Saved: {calib_save_path}")
        
        benchmarks.append({
            'file': filename,
            'input': f"{w}x{h}",
            'output': f"{out_w}x{out_h}",
            'faces': num_faces,
            'time_baseline': elapsed_baseline,
            'time_calibrated': elapsed_calib,
            'baseline_file': baseline_save_path,
            'calib_file': calib_save_path
        })
    
    print("\n" + "="*70)
    print("FINAL BENCHMARK REPORT")
    print("="*70)
    for b in benchmarks:
        print(f"Image: {b['file']:20s} | In: {b['input']:10s} | Out: {b['output']:10s} | Faces: {b['faces']} | Base: {b['time_baseline']:.2f}s | Calib: {b['time_calibrated']:.2f}s")

if __name__ == '__main__':
    main()
