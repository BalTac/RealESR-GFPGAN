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

### 3.1 Protocol

Eight publicly available degraded samples were used — seven with faces, one greyscale
scene without faces which serves as a control (`bicycle_bw` has no face, so any face
stage must leave it untouched):

| sample | source | size | note |
|---|---|---|---|
| `00003` | Real-ESRGAN demo inputs | 512×256 | composite of heavily degraded faces |
| `0014` | Real-ESRGAN demo inputs | 179×179 | very small, severe JPEG damage |
| `0030` | Real-ESRGAN demo inputs | 220×220 | very small, oversharpened input |
| `OST_009` | Real-ESRGAN demo inputs | 448×640 | real photograph, motion/compression damage |
| `10045` | GFPGAN demo inputs | 783×1146 | group scene, three distant faces |
| `00` | GFPGAN demo inputs | 1297×1920 | vintage portrait, chemical noise |
| `Blake_Lively` | GFPGAN demo inputs | 700×464 | mildly degraded, the easiest case |
| `bicycle_bw` | this repository | 768×768 | greyscale, no faces (control) |

Metrics: NIQE and BRISQUE (both lower-better, `pyiqa` 0.1.16 implementations) plus MUSIQ
(higher-better, learned quality model). Every output is resized so its longest side is
1024 px **before** scoring: NIQE and BRISQUE are resolution dependent, and comparing a 2×
output against a 4× output without normalising would measure the scale instead of the
restoration. Reproduction: `python scripts/eval_metrics.py`, raw values in
`results/metrics.json`.

Variants:

* **input** — untouched file, no processing;
* **GFPGAN only** — bicubic 2× + GFPGAN (w = 0.7), no scene super-resolution;
* **ESRGAN only** — `RealESRGAN_x2plus`, no face stage;
* **v1 fusion** — w = 0.60, α = 0.20, β = 1.0 (the original configuration);
* **v2 fusion** — w = 0.70, α = 0.30, β = 0.80.

Note on the numbers previously reported for v1 (NIQE 3.61–4.02): they are **not
reproducible** with this protocol and have been removed. The tables below are what the
shipped code produces today.

### 3.2 NIQE (lower is better)

| Image | input | GFPGAN only | ESRGAN only | v1 fusion | v2 fusion |
|---|---|---|---|---|---|
| 00003 | 3.92 | 5.07 | 4.50 | 4.49 | 4.49 |
| 0014 | 6.87 | 6.87 | 5.04 | 4.90 | 4.63 |
| 0030 | 5.24 | 6.97 | 6.10 | 5.84 | 6.40 |
| OST_009 | 5.52 | 4.70 | 4.65 | 4.87 | 5.18 |
| 10045 | 3.83 | 3.85 | 4.26 | 3.42 | 3.57 |
| 00 | 3.22 | 3.11 | 3.42 | 3.24 | 3.29 |
| Blake_Lively | 3.89 | 3.85 | 4.01 | 3.82 | 3.93 |
| bicycle_bw | 2.80 | 3.97 | 2.64 | 2.64 | 2.65 |
| **mean** | **4.41** | **4.80** | **4.33** | **4.15** | **4.27** |

### 3.3 BRISQUE (lower is better)

| Image | input | GFPGAN only | ESRGAN only | v1 fusion | v2 fusion |
|---|---|---|---|---|---|
| 00003 | −7.06 | 34.58 | 7.46 | 7.64 | 7.73 |
| 0014 | 17.76 | 57.00 | 13.73 | 35.06 | 32.34 |
| 0030 | 7.29 | 55.74 | 4.72 | 27.81 | 26.67 |
| OST_009 | 49.22 | 31.38 | 11.16 | 34.11 | 34.01 |
| 10045 | 21.71 | 20.83 | 23.49 | 13.61 | 14.68 |
| 00 | 18.25 | 16.08 | 29.47 | 24.99 | 25.13 |
| Blake_Lively | 17.09 | 8.65 | 7.46 | 2.42 | 2.74 |
| bicycle_bw | 29.55 | 34.71 | 28.35 | 28.35 | 28.35 |
| **mean** | **19.23** | **32.37** | **15.73** | **21.75** | **21.46** |

### 3.4 MUSIQ (higher is better)

| Image | input | GFPGAN only | ESRGAN only | v1 fusion | v2 fusion |
|---|---|---|---|---|---|
| 00003 | 45.01 | 32.97 | 52.96 | 51.86 | 51.51 |
| 0014 | 37.63 | 25.28 | 74.80 | 44.18 | 44.49 |
| 0030 | 52.41 | 34.54 | 76.13 | 53.72 | 56.01 |
| OST_009 | 72.54 | 65.87 | 73.91 | 66.89 | 67.86 |
| 10045 | 34.78 | 40.51 | 54.47 | 45.81 | 45.55 |
| 00 | 67.48 | 72.05 | 72.72 | 68.12 | 70.32 |
| Blake_Lively | 67.10 | 76.01 | 76.51 | 72.31 | 71.10 |
| bicycle_bw | 70.80 | 64.17 | 71.91 | 71.91 | 71.89 |
| **mean** | **55.97** | **51.43** | **69.18** | **59.35** | **59.84** |

### 3.5 What the numbers actually say

Three conclusions, one of which is a negative result:

1. **The facial prior alone is the weakest option.** GFPGAN without a scene
   super-resolver has the worst BRISQUE mean (32.37 vs 15.73 for the plain backbone) and
   the worst MUSIQ mean (51.43 vs 69.18). This is the quantitative form of the "plastic
   skin" premise: on damaged inputs the generator buys sharpness at the cost of
   plausibility, and the metrics notice. Fusing it with the background recovers most of
   that loss (BRISQUE 21.5 instead of 32.4).
2. **Fusion helps NIQE, costs MUSIQ.** v1 fusion has the best NIQE mean (4.15 against
   4.33 for the backbone alone): it smooths away some of the noise and halftone damage
   that NIQE penalises. MUSIQ, which rewards crisp detail, prefers the plain backbone on
   all eight samples. The two metrics disagree because they reward opposite things;
   neither is a proxy for "looks right".
3. **The difference between v1 and v2 fusion is small and not consistent in sign.**
   v1 wins NIQE on 6 of 8, v2 wins BRISQUE on 5 of 8, MUSIQ is effectively a tie. The
   measured difference between β = 1.0 and β = 0.8 sits inside the noise of these
   metrics — which is why β is exposed as a flag instead of being presented as a quality
   improvement. The texture measurement (Laplacian variance 27.7 → 21.3 in the face
   region) describes what β does mechanically; it does not prove it looks better.

The control sample confirms there is no hidden effect: `bicycle_bw` has no face, so
"ESRGAN only", "v1 fusion" and "v2 fusion" produce the same image and the same scores.

**Caveat.** These are no-reference metrics on eight samples from two demo sets, one
hardware configuration, one run per image. They are sensitive to input composition:
`00003` is a composite whose BRISQUE score is negative before any processing. Treat the
tables as a sanity check on the pipeline, not as evidence of state-of-the-art quality.

### 3.6 Timings and texture (same machine)

| Measurement | Result |
|---|---|
| 2× output via native 2× backbone vs 4× + downscale | upscale 26.6 s → 7.3 s, end-to-end 37.9 s → 12.7 s (1297×1920) |
| fp16 autocast vs fp32, same input | 26 s → 108 s (4× slower) on GTX 1660 Ti (Turing TU116, no tensor cores) |
| tiled vs single-pass inference | mean absolute difference 1.70/255 |
| face region, β = 1.0 vs β = 0.8 | Laplacian variance 27.7 → 21.3, high-frequency std 4.60 → 3.93 |
| plain upscale, no face pass (reference) | Laplacian variance 4.2, high-frequency std 1.47 |
| loading backbones from a v2 bundle vs separate files | byte-identical output, mean difference 0 |

### 3.7 Checkpoints

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

### 3.8 Hardware

Single machine, single GPU: NVIDIA GTX 1660 Ti, 6 GB VRAM, Turing TU116 (no tensor
cores), torch 2.11 + cu128, Windows. Peak allocation during a 2× restore of a 1297×1920
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
* No-reference metrics are reported in 3.2-3.5, but they disagree with each other and
  are sensitive to input composition: they are a sanity check, not a leaderboard. No
  full-reference benchmark (PSNR/SSIM/LPIPS on aligned pairs) was run, because no
  ground-truth pairs ship with this repository.

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
