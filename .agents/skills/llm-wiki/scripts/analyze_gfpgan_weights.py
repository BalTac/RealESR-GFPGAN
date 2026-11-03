import torch

ckpt_path = 'experiments/pretrained_models/GFPGANv1.4.pth'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ckpt['params_ema']

modules = {}
total_params = 0
for k, v in sd.items():
    prefix = k.split('.')[0]
    numel = v.numel()
    total_params += numel
    modules.setdefault(prefix, {'count': 0, 'params': 0, 'first': k, 'last': k, 'examples': []})
    modules[prefix]['count'] += 1
    modules[prefix]['params'] += numel
    modules[prefix]['last'] = k
    if len(modules[prefix]['examples']) < 3:
        modules[prefix]['examples'].append((k, tuple(v.shape), float(v.mean()), float(v.std())))

print(f"=== GFPGAN v1.4 WEIGHT-SPACE ANALYSIS ===")
print(f"Total parameter tensors : {len(sd)}")
print(f"Total parameter count   : {total_params:,} ({total_params/1e6:.2f}M float32 params)")

print("\n--- Sub-Module Breakdown ---")
for mod, info in sorted(modules.items(), key=lambda x: -x[1]['params']):
    pct = (info['params'] / total_params) * 100
    print(f"{mod:25s} | {info['count']:3d} tensors | {info['params']:10,d} params ({pct:5.1f}%) | {info['first']} -> {info['last']}")

print("\n--- Structural Analysis by Component ---")
# 1. Degradation Face Encoder (conv_body)
conv_body = {k: v for k, v in sd.items() if k.startswith('conv_body')}
print(f"1. Degradation Face Encoder (U-Net/ResNet backbone): {len(conv_body)} tensors, {sum(v.numel() for v in conv_body.values()):,} params")
print(f"   Input shape: conv_body_first.weight -> {tuple(conv_body['conv_body_first.weight'].shape)} (3 RGB input channels)")

# 2. StyleGAN2 Prior Generator (stylegan_decoder)
stylegan = {k: v for k, v in sd.items() if k.startswith('stylegan_decoder')}
print(f"2. Facial Prior Generator (StyleGAN2 Pretrained Core): {len(stylegan)} tensors, {sum(v.numel() for v in stylegan.values()):,} params")

# 3. Spatial Feature Transform (SFT) modulation
sft = {k: v for k, v in sd.items() if 'sft' in k.lower() or 'condition' in k.lower()}
print(f"3. SFT / Modulation Connectors: {len(sft)} tensors, {sum(v.numel() for v in sft.values()):,} params")

# 4. Latent mapping layers
latents = {k: v for k, v in sd.items() if 'latent' in k.lower() or 'style' in k.lower()}
print(f"4. Style / Latent Vectors: {len(latents)} tensors, {sum(v.numel() for v in latents.values()):,} params")

# Weight distribution check
weights_flat = torch.cat([v.flatten() for v in list(sd.values())[:30]])
print(f"\nSample Weight Distribution (First 30 layers): Mean={weights_flat.mean():.6f}, Std={weights_flat.std():.6f}, Min={weights_flat.min():.6f}, Max={weights_flat.max():.6f}")
