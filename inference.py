import argparse
import os
import time
import cv2
import torch
from realesr_gfpgan import RealESR_GFPGAN

def main():
    parser = argparse.ArgumentParser(description="RealESR-GFPGAN Unified Image Restoration")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, default="results/output.jpg", help="Path to output image")
    parser.add_argument("-m", "--model", type=str, default="experiments/pretrained_models/RealESR_GFPGAN_v1.pth", help="Path to unified .pth checkpoint")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Upscale factor (default: 2)")
    parser.add_argument("-w", "--weight", type=float, default=0.60, help="Fidelity weight (default: 0.60)")
    parser.add_argument("--tile", type=int, default=400, help="Tile size for RealESRGAN (default: 400)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Inference] Initializing RealESR-GFPGAN on {device}...")
    
    model = RealESR_GFPGAN(
        checkpoint_path=args.model,
        upscale=args.scale,
        tile=args.tile,
        device=device
    )

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    print(f"[Inference] Processing {args.input} ({w}x{h} px)...")

    t0 = time.time()
    restored = model.enhance(img, fidelity_weight=args.weight)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, restored)
    out_h, out_w = restored.shape[:2]
    print(f"[Inference] Saved to {args.output} ({out_w}x{out_h} px) in {elapsed:.2f}s!")

if __name__ == '__main__':
    main()
