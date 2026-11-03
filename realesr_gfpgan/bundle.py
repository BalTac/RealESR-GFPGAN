import os
import torch

def create_unified_checkpoint(
    esrgan_path='experiments/pretrained_models/RealESRGAN_x2plus.pth',
    gfpgan_path='experiments/pretrained_models/GFPGANv1.4.pth',
    output_path='experiments/pretrained_models/RealESR_GFPGAN_v1.pth'
):
    print(f"Bundling RealESRGAN ({esrgan_path}) and GFPGAN ({gfpgan_path})...")
    
    ckpt_esrgan = torch.load(esrgan_path, map_location='cpu')
    key_esr = 'params_ema' if 'params_ema' in ckpt_esrgan else 'params'
    state_esr = ckpt_esrgan[key_esr] if key_esr in ckpt_esrgan else ckpt_esrgan
    
    ckpt_gfp = torch.load(gfpgan_path, map_location='cpu')
    key_gfp = 'params_ema' if 'params_ema' in ckpt_gfp else 'params'
    state_gfp = ckpt_gfp[key_gfp] if key_gfp in ckpt_gfp else ckpt_gfp
    
    bundle = {
        'version': '1.0.0',
        'architecture': 'RealESR_GFPGAN_Unified',
        'config': {
            'upscale': 2,
            'fidelity_weight': 0.60,
            'micro_texture_alpha': 0.20,
            'blend_radius': 25,
            'tile': 400,
            'tile_pad': 16
        },
        'params_esrgan': state_esr,
        'params_gfpgan': state_gfp
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(bundle, output_path)
    print(f"Unified checkpoint successfully created at: {output_path}")
    print(f"Total bundle size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    create_unified_checkpoint()
