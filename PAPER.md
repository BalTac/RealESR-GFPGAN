# RealESR-GFPGAN v2: Fusing Residual Dense Convolutions with Generative Facial Priors for Artifact-Free Scene Restoration

**Technical Report & Architecture Specification**
*Deep Learning Image Restoration Group (2026)*

> **Scope of this revision.** v1 described a single hard-wired pipeline: one 23-block
> RRDBNet, one bundled checkpoint, one fixed set of weights. v2 keeps the same idea
> (scene super-resolution + facial prior + boundary-aware fusion) but makes the
> backbone pluggable, removes the `basicsr` dependency, and ships two checkpoints with
> separate licences. Section 3 distinguishes **v1 numbers, not re-measured** from
> **v2 measurements taken on the machine described in 3.4**.

---

## 1. Introduction

Blind image restoration in the wild presents a fundamental trade-off:

1. **Geometric/Texture Prior Blindness:** Generic super-resolution networks such as
   Real-ESRGAN (RRDBNet) optimize structural and perceptual losses across the whole
   field of view, but do not hallucinate missing semantic features on deeply degraded
   faces (iris reflections, eyelashes, dental boundaries).
2. **Facial Over-Hallucination & Background Artifacts:** Specialized facial priors such
   as GFPGAN v1.4 (StyleGAN2) and CodeFormer (VQGAN codebooks) restore crisp facial
   anatomy. However:
   * **StyleGAN2 latent bias:** at moderate fidelity weights the prior interpolates
     through smooth synthetic manifolds, producing waxy skin without pores or grain.
   * **Seam discontinuities:** the restorer crops a face box and pastes it back over a
     naively upsampled background, leaving visible boundaries.

v2 addresses both, and adds what v1 lacked: the super-resolution backbone is a
**parameter** (any ESRGAN/RRDB checkpoint that `spandrel` can identify), the fusion has
a tunable **face/background mix**, and the deployment artifact is a checkpoint with
named variants instead of a single opaque file.

---

## 2. Methodology

```
                    Input image I_in (BGR uint8)
                              |
              +---------------+---------------+
              |                               |
              v                               v
   +----------------------+        +----------------------+
   |  Backbone (spandrel) |        |  GFPGAN v1.4         |
   |  RRDB / ESRGAN,      |        |  RetinaFace detect,  |
   |  scale s in {2,4}    |        |  StyleGAN2 decode    |
   |  tiled + overlap acc |        |  fidelity w          |
   +----------+-----------+        +----------+-----------+
              |                               |
              v  I_bg                          v  I_face (aligned box pasted back)
   +----------------------+        +----------------------+
   |                      |        |  Unsharp calibration |
   |                      |        |  alpha = 0.30        |
   |                      |        +----------+-----------+
   |                      |                   |
   |                      |                   v  I_calib
   |                      +---------+---------+
   |                                |
   v                                v
   +------------------------------------------------+
   |  M = Gauss25(Dilate15x2( [ |I_calib - I_ref| > tau ] ))
   |  I_out = M * (beta*I_calib + (1-beta)*I_bg) + (1-M)*I_bg
   +------------------------------------------------+
```

### 2.1 Backbone super-resolution (pluggable)

The scene is upscaled by any checkpoint `spandrel` can load; the RRDBNet family used in
v1 is still supported (23 RRDB blocks, `num_feat=64`, `num_grow_ch=32`), but so are
ESRGAN variants with different depths and native scales. The native scale of the loaded
model is read from the model descriptor, and the requested output scale is reached by
Lanczos resampling only when they differ:

```
s_model = desc.scale            # 2 for RealESRGAN_x2plus, 4 for RealESRGAN_x4plus
I_bg    = TileStitch(NN(I_in), P=384, pad=32)
I_out   = s_model == s_req ? I_bg : Lanczos(I_bg, s_req / s_model)
```

Tiles advance by `P - 2*pad`, and overlapping predictions are **averaged through an
accumulator** rather than written with hard seams:

```
I_bg(p) = sum_t NN(tile_t)(p) / sum_t 1[p in tile_t]
```

### 2.2 Facial prior and anti-plastic calibration

Landmarks are detected with RetinaFace-ResNet50, faces are aligned, and restored by the
GFPGAN StyleGAN2 decoder with fidelity weight `w`:

```
I_face = G_StyleGAN2(E(I_aligned), w)          w = 0.70 (default v2), 0.60 (v1)
I_calib = I_face + alpha * (I_face - G_{sigma=1.2}(I_face))     alpha = 0.30
```

The unsharp term re-injects micro-contrast that the latent prior smooths away.

### 2.3 Boundary-aware fusion with face/background mix

```
D   = |I_calib - I_ref|                        I_ref = Lanczos(I_in, size(I_bg))
M0  = 1[D > tau]                               tau = 15
M1  = dilate(M0, 15x15, iterations=2)
M   = gaussian(M1, 25x25)
```

v2 adds a mixing coefficient inside the mask, which keeps some of the upscaler's grain
instead of the pure generator output:

```
I_out = M * (beta*I_calib + (1-beta)*I_bg) + (1-M)*I_bg        beta = 0.80
```

`beta = 1` reproduces v1 exactly; `beta = 0` is a pure super-resolution result.

### 2.4 Precision

fp16 autocast is available (`--half`) but **off by default**. On GPUs without tensor
cores the mixed-precision path is a regression, measured in 3.4.

---

## 3. Evaluation

### 3.1 v1 numbers (as reported, **not re-measured on v2**)

These NIQE/BRISQUE values were produced during the v1 experiments on the hardware
available at the time. They are kept for historical comparison and were **not**
reproduced here: v2 uses different backbones, different defaults and a different
fusion, so the two rows are not directly comparable.

| Scenario | Input NIQE | Baseline GFPGAN | CodeFormer (w=0.6) | v1 RealESR-GFPGAN |
| :--- | :---: | :---: | :---: | :---: |
| Blake Lively (700×464) | 7.12 | 4.82 | 4.15 | 3.61 |
| Group 10045 (546×800) | 6.89 | 5.34 | 4.48 | 3.89 |
| Vintage 00 (540×800) | 8.45 | 5.91 | 4.70 | 4.02 |
| Mean | 7.49 | 5.36 | 4.44 | 3.84 |

### 3.2 v2 measurements

Test set: one 640×480 scan from 2003 (one face, JPEG artifacts), one 768×768
black-and-white photograph, one 1297×1920 portrait (two faces). No public benchmark
suite was re-run, so no PSNR/NIQE claim is made for v2.

| Measurement | Result |
|---|---|
| 2x output via native 2x backbone (`RealESRGAN_x2plus`) vs 4x + downscale | upscale 26.6 s → 7.3 s, end-to-end 37.9 s → 12.7 s (1297×1920) |
| fp16 autocast vs fp32, same input | 26 s → 108 s (4× slower) on GTX 1660 Ti (Turing TU116, no tensor cores) |
| tiled vs single-pass inference | mean absolute difference 1.70/255 |
| face region with `beta=1` (v1 fusion) vs `beta=0.8` | Laplacian variance 27.7 → 21.3, high-frequency std 4.60 → 3.93 |
| plain upscale, no face pass (reference) | Laplacian variance 4.2, high-frequency std 1.47 |
| loading backbones from a v2 bundle vs separate files | byte-identical output, mean difference 0 |

Interpretation: `beta = 0.8` does not increase sharpness — it **reduces synthetic
high-frequency energy** by blending in the upscaler's texture. Whether that is better is
a judgement call; the flag exists so the trade-off is explicit rather than baked in.

### 3.3 Checkpoints

v2 ships two release assets. The bundling code repacks third-party weights without
modifying their values and records the architecture name per variant:

```
{
  "version": "2.0.0",
  "architecture": "RealESR_GFPGAN_v2",
  "variants":      {"x2": <state_dict>, "x4": <state_dict>},
  "esrgan_arch":   {"x2": "ESRGAN",     "x4": "ESRGAN"},
  "params_gfpgan": <state_dict>,
  "config": {"default_variant": "x2", "fidelity_weight": 0.70,
             "micro_texture_alpha": 0.30, "face_mix": 0.80,
             "mask_threshold": 15, "tile": 384, "tile_pad": 32, "half": false}
}
```

| Asset | Backbones | Size | Licence | Commercial use |
|---|---|---|---|---|
| `RealESR_GFPGAN_v2.pth` | `RealESRGAN_x2plus` (x2), `RealESRGAN_x4plus` (x4), `GFPGANv1.4` | 460 MB | BSD-3 + Apache-2.0 | allowed |
| `RealESR_GFPGAN_v2_ultrasharp.pth` | `4x-UltraSharp` (x4), `GFPGANv1.4` | 396 MB | CC BY-NC-SA 4.0 + Apache-2.0 | **not allowed** |

The v1 single-file format (`params_esrgan`/`params_gfpgan`) is still readable. The
repository code is Apache-2.0 and is not a derivative work of the weights: the
non-commercial restriction applies to the bundle that carries UltraSharp, not to the
program that loads it. Full attribution in `NOTICE`, licence text in
`LICENSE-CC-BY-NC-SA-4.0`.

### 3.4 Hardware

Single machine, single GPU: NVIDIA GTX 1660 Ti, 6 GB VRAM, Turing TU116 (no tensor
cores), torch 2.11 + cu128, Windows. Peak allocation during a 2x restore of a 1297×1920
input: ~1.14 GB allocated, ~1.36 GB reserved. Inference is process-per-request, so the
model is loaded and freed on every call; no resident server.

---

## 4. Limitations

* The face prior is generative: on heavily degraded inputs identity drift and "other
  person" effects remain possible. `beta` and `w` trade sharpness for fidelity, not
  identity correctness.
* Hands, jewellery and small objects are not covered by any stage; they stay soft.
  Hair can take on a "silk" look from the generator.
* Colourisation of black-and-white input is out of scope. Lab-space colorisers
  (`eccv16`, `siggraph17`) preserve geometry by construction but guess hue; prompt-guided
  img2img gives the requested colour but changes geometry unless a ControlNet pins the
  contours — no ControlNet is bundled here.
* Colour-managed inputs and non-8-bit sources are not handled; the pipeline works on
  BGR `uint8`.
* No quantitative no-reference benchmark was run for v2. The numbers in 3.2 are
  machine-specific timings and local texture statistics, not perceptual quality scores.

---

## 5. Conclusion

v2 keeps the composition that worked — super-resolved scene, restored face, soft mask
fusion — and removes what did not: a hard dependency on an unmaintained library, a
single opaque checkpoint, and defaults tuned for one workstation. The result is a
pipeline where the backbone is a parameter, the checkpoint carries its own metadata, the
fusion is a documented coefficient, and every claim in this report is either a v1 number
labelled as such or a measurement reproducible with `scripts/selfcheck.py` and
`scripts/bench.py`.

The weight calibration strategy is inspired by the principles of **Weight Space
Programming** introduced by Rikiya Takahashi [1].

---

## References

1. **Rikiya Takahashi**, *"Weight Space Programming"*, 2024. https://rikiyat.github.io/blog/weight-space-programming/
2. **Xintao Wang, Yu Li, Honglun Zhang, Ying Shan**, *"Towards Real-World Blind Face Restoration with Generative Facial Prior"*, CVPR 2021. (GFPGAN)
3. **Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan**, *"Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"*, ICCVW 2021.
4. **Shangchen Zhou, Kelvin C.K. Chan, Chongyi Li, Chen Change Loy**, *"Towards Robust Blind Face Restoration with Codebook Lookup Transformer"*, NeurIPS 2022. (CodeFormer)
5. **Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, Stefanos Zafeiriou**, *"RetinaFace: Single-shot Multi-level Face Localisation in the Wild"*, CVPR 2020.
6. **Kim2091**, *4x-UltraSharp*, CC BY-NC-SA 4.0. https://huggingface.co/Kim2091/UltraSharp
7. **spandrel** (model loader), MIT. https://github.com/chaiNNer-org/spandrel
8. **facexlib** (detection/parsing utilities), MIT.
