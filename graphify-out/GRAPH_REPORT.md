# Graph Report - gfpgan  (2026-11-03)

## Corpus Check
- 111 files · ~85,917 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 11 file(s) not represented in the graph (top: .pth 6, (none) 5)

## Summary
- 1847 nodes · 3332 edges · 137 communities (97 shown, 40 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 77 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- data_util.py
- check_evidence.py
- bisenet.py
- VideoReader
- torch
- numpy
- train
- .find_class
- train
- .__getitem__
- train
- yolo.py
- train
- deform_conv_cuda.cpp
- train
- cv2
- ffhq_blind_joint_dataset.py
- matlab_cp2tform.py
- facelib/utils/misc.py
- PairedImageDataset
- arcface_arch.py
- arch_util.py
- retinaface.py
- retinaface_utils.py
- train
- face_detector.py
- train
- file_client.py
- BaseModel
- UpFirDn2DKernelParams
- train
- matlab_functions.py
- ShuffleV2Block
- train
- deform_conv.py
- CodeFormerIdxModel
- dist_util.py
- base_model.py
- RetinaFace
- Model
- default_init_weights
- prefetch_dataloader.py
- train_pipeline
- psnr_ssim.py
- get_root_logger
- L1Loss
- setup.py
- crop_align_face.py
- train
- upfirdn2d_kernel.cu
- losses.py
- Registry
- GANLoss
- VQGANModel
- .__getitem__
- ResBlock
- vgg_arch.py
- .__init__
- DeformConvFunction
- time
- detection/__init__.py
- .batched_detect_faces
- CodeFormer_colorization.yml
- CodeFormer_inpainting.yml
- CodeFormer_stage2.yml
- CodeFormer_stage3.yml
- VQGAN_512_ds32_nearest_stage1.yml
- AttnBlock
- weighted_loss
- CodeFormerJointModel
- CodeFormerModel
- SRModel
- DeformConvPack
- network_g
- network_g
- network_g
- FileClient
- fused_bias_act.cpp
- upfirdn2d.cpp
- psnr
- psnr
- psnr
- psnr
- network_g
- psnr
- TransformerSALayer
- EnlargedSampler
- Bottleneck
- yolov5l.yaml
- yolov5n.yaml
- logger
- network_vqgan
- path
- logger
- network_vqgan
- path
- logger
- network_vqgan
- path
- logger
- network_vqgan
- logger
- network_g
- path
- CodeFormer
- VQAutoEncoder
- CUDAPrefetcher
- CharbonnierLoss
- PrefetchReader
- network_d
- path
- adaptive_instance_normalization
- LmdbMaker
- master_only
- optim_g
- network_d
- PositionEmbeddingSine
- Detect
- scheduler
- optim_g
- optim_d
- optim_g
- ClassHead
- LandmarkHead
- perceptual_opt
- io_backend
- scheduler
- register
- scheduler
- pixel_opt
- pixel_unshuffle

## God Nodes (most connected - your core abstractions)
1. `get_root_logger()` - 56 edges
2. `train` - 30 edges
3. `train` - 30 edges
4. `tensor2img()` - 27 edges
5. `BaseModel` - 26 edges
6. `train` - 26 edges
7. `imwrite()` - 24 edges
8. `SRModel` - 23 edges
9. `UpFirDn2DKernelParams` - 21 edges
10. `img2tensor()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `run()` --uses--> `FaceRestoreHelper`  [INFERRED]
  .agents/skills/llm-wiki/scripts/run_codeformer.py → CodeFormer_src/CodeFormer-master/facelib/utils/face_restoration_helper.py
- `run()` --calls--> `tensor2img()`  [INFERRED]
  .agents/skills/llm-wiki/scripts/run_codeformer.py → CodeFormer_src/CodeFormer-master/basicsr/utils/img_util.py
- `main()` --uses--> `RRDBNet`  [INFERRED]
  .agents/skills/llm-wiki/scripts/run_hybrid.py → CodeFormer_src/CodeFormer-master/basicsr/archs/rrdbnet_arch.py
- `run()` --calls--> `normalize()`  [INFERRED]
  .agents/skills/llm-wiki/scripts/run_codeformer.py → .agents/skills/llm-wiki/scripts/check_evidence.py
- `run()` --calls--> `get_device()`  [EXTRACTED]
  .agents/skills/llm-wiki/scripts/run_codeformer.py → CodeFormer_src/CodeFormer-master/basicsr/utils/misc.py

## Import Cycles
- 2-file cycle: `CodeFormer_src/CodeFormer-master/basicsr/ops/fused_act/__init__.py -> CodeFormer_src/CodeFormer-master/basicsr/ops/fused_act/fused_act.py -> CodeFormer_src/CodeFormer-master/basicsr/ops/fused_act/__init__.py`
- 2-file cycle: `CodeFormer_src/CodeFormer-master/basicsr/ops/dcn/__init__.py -> CodeFormer_src/CodeFormer-master/basicsr/ops/dcn/deform_conv.py -> CodeFormer_src/CodeFormer-master/basicsr/ops/dcn/__init__.py`
- 3-file cycle: `CodeFormer_src/CodeFormer-master/facelib/detection/__init__.py -> CodeFormer_src/CodeFormer-master/facelib/utils/__init__.py -> CodeFormer_src/CodeFormer-master/facelib/utils/face_utils.py -> CodeFormer_src/CodeFormer-master/facelib/detection/__init__.py`
- 4-file cycle: `CodeFormer_src/CodeFormer-master/facelib/detection/__init__.py -> CodeFormer_src/CodeFormer-master/facelib/utils/__init__.py -> CodeFormer_src/CodeFormer-master/facelib/utils/face_utils.py -> CodeFormer_src/CodeFormer-master/facelib/utils/face_restoration_helper.py -> CodeFormer_src/CodeFormer-master/facelib/detection/__init__.py`
- 4-file cycle: `CodeFormer_src/CodeFormer-master/facelib/parsing/__init__.py -> CodeFormer_src/CodeFormer-master/facelib/utils/__init__.py -> CodeFormer_src/CodeFormer-master/facelib/utils/face_utils.py -> CodeFormer_src/CodeFormer-master/facelib/utils/face_restoration_helper.py -> CodeFormer_src/CodeFormer-master/facelib/parsing/__init__.py`

## Communities (137 total, 40 thin omitted)

### Community 0 - "data_util.py"
Cohesion: 0.12
Nodes (9): brush_stroke_mask(), duf_downsample(), generate_frame_indices(), generate_gaussian_kernel(), paths_from_lmdb(), random_ff_mask(), read_img_seq(), mod_crop() (+1 more)

### Community 1 - "check_evidence.py"
Cohesion: 0.12
Nodes (22): Candidate, check_article(), contains(), Document, extract_candidates(), flush_blockquote(), flush_paragraph(), extract_numeric_date_candidates() (+14 more)

### Community 2 - "bisenet.py"
Cohesion: 0.06
Nodes (16): AttentionRefinementModule, BiSeNet, BiSeNetOutput, ContextPath, ConvBNReLU, FeatureFusionModule, init_parsing_model(), ConvLayer (+8 more)

### Community 3 - "VideoReader"
Cohesion: 0.12
Nodes (3): get_video_meta_info(), VideoReader, VideoWriter

### Community 4 - "torch"
Cohesion: 0.05
Nodes (29): run(), enhance_background_deconvolution(), main(), main(), run_realesrgan_upscale(), RRDBNet, build_loss(), calculate_metric() (+21 more)

### Community 5 - "numpy"
Cohesion: 0.11
Nodes (21): bivariate_anisotropic_Gaussian(), bivariate_generalized_Gaussian(), bivariate_isotropic_Gaussian(), bivariate_plateau_type1(), bivariate_plateau_type1_iso(), bivariate_skew_Gaussian(), bivariate_skew_Gaussian_center(), cdf2() (+13 more)

### Community 6 - "train"
Cohesion: 0.05
Nodes (42): gan_type, loss_weight, type, betas, lr, type, weight_decay, betas (+34 more)

### Community 8 - "train"
Cohesion: 0.05
Nodes (42): gan_type, loss_weight, type, betas, lr, type, weight_decay, betas (+34 more)

### Community 10 - "train"
Cohesion: 0.17
Nodes (12): gan_type, loss_weight, type, train, ema_decay, gan_opt, manual_seed, net_d_iters (+4 more)

### Community 11 - "yolo.py"
Cohesion: 0.12
Nodes (9): autopad(), Conv, DWConv(), Focus, SPP, StemBlock, CrossConv, MixConv2d (+1 more)

### Community 12 - "train"
Cohesion: 0.06
Nodes (31): train, type, batch_size_per_gpu, blur_kernel_size, blur_sigma, color_jitter_prob, color_jitter_pt_prob, color_jitter_shift (+23 more)

### Community 13 - "deform_conv_cuda.cpp"
Cohesion: 0.14
Nodes (18): deform_conv_backward_input_cuda(), deform_conv_backward_parameters_cuda(), deform_conv_forward_cuda(), deformable_col2im(), deformable_col2im_coord(), deformable_im2col(), modulated_deformable_col2im_coord_cuda(), modulated_deformable_col2im_cuda() (+10 more)

### Community 14 - "train"
Cohesion: 0.07
Nodes (29): train, batch_size_per_gpu, blur_kernel_size, blur_sigma, blur_sigma_large, dataroot_gt, dataset_enlarge_ratio, downsample_range (+21 more)

### Community 15 - "cv2"
Cohesion: 0.18
Nodes (4): img_rotate(), tensor2img_fast(), make_lmdb_from_imgs(), read_img_worker()

### Community 17 - "matlab_cp2tform.py"
Cohesion: 0.11
Nodes (12): FaceWarpException, get_affine_transform_matrix(), get_reference_facial_points(), warp_and_crop_face(), cvt_tform_mat_for_cv2(), findNonreflectiveSimilarity(), findSimilarity(), get_similarity_transform() (+4 more)

### Community 18 - "facelib/utils/misc.py"
Cohesion: 0.05
Nodes (18): normalize(), FaceRestoreHelper, get_center_face(), get_largest_face(), align_crop_face_landmarks(), compute_increased_bbox(), get_valid_bboxes(), paste_face_back() (+10 more)

### Community 19 - "PairedImageDataset"
Cohesion: 0.18
Nodes (4): paired_paths_from_folder(), paired_paths_from_lmdb(), paired_paths_from_meta_info_file(), PairedImageDataset

### Community 20 - "arcface_arch.py"
Cohesion: 0.09
Nodes (6): BasicBlock, Bottleneck, conv3x3(), IRBlock, ResNetArcFace, SEBlock

### Community 21 - "arch_util.py"
Cohesion: 0.10
Nodes (7): flow_warp(), make_layer(), _no_grad_trunc_normal_(), _ntuple(), resize_flow(), trunc_normal_(), Upsample

### Community 22 - "retinaface.py"
Cohesion: 0.17
Nodes (12): generate_config(), BboxHead, conv_bn(), conv_bn1X1(), conv_bn_no_relu(), conv_dw(), FPN, make_bbox_head() (+4 more)

### Community 23 - "retinaface_utils.py"
Cohesion: 0.10
Nodes (11): center_size(), encode(), encode_landm(), intersect(), jaccard(), log_sum_exp(), match(), matrix_iof() (+3 more)

### Community 24 - "train"
Cohesion: 0.07
Nodes (27): train, type, batch_size_per_gpu, blur_kernel_size, blur_sigma, dataroot_gt, dataset_enlarge_ratio, downsample_range (+19 more)

### Community 25 - "face_detector.py"
Cohesion: 0.12
Nodes (13): isListempty(), Detections, letterbox(), box_iou(), check_img_size(), clip_coords(), make_divisible(), non_max_suppression() (+5 more)

### Community 26 - "train"
Cohesion: 0.14
Nodes (14): train, cross_entropy_loss, ema_decay, entropy_loss_weight, feat_loss_weight, fidelity_weight, manual_seed, net_d_iters (+6 more)

### Community 27 - "file_client.py"
Cohesion: 0.11
Nodes (4): BaseStorageBackend, HardDiskBackend, LmdbBackend, MemcachedBackend

### Community 29 - "UpFirDn2DKernelParams"
Cohesion: 0.11
Nodes (19): UpFirDn2DKernelParams, down_x, down_y, in_h, in_w, kernel_h, kernel_w, loop_major (+11 more)

### Community 30 - "train"
Cohesion: 0.11
Nodes (19): train, type, batch_size_per_gpu, dataroot_gt, dataset_enlarge_ratio, filename_tmpl, gen_inpaint_mask, gt_size (+11 more)

### Community 31 - "matlab_functions.py"
Cohesion: 0.16
Nodes (8): calculate_weights_indices(), _convert_input_type_range(), _convert_output_type_range(), cubic(), imresize(), rgb2ycbcr(), ycbcr2bgr(), ycbcr2rgb()

### Community 33 - "train"
Cohesion: 0.11
Nodes (18): train, type, batch_size_per_gpu, dataroot_gt, dataset_enlarge_ratio, filename_tmpl, gt_size, in_size (+10 more)

### Community 34 - "deform_conv.py"
Cohesion: 0.23
Nodes (3): DCNv2Pack, ModulatedDeformConv, ModulatedDeformConvPack

### Community 37 - "dist_util.py"
Cohesion: 0.38
Nodes (5): parse_options(), get_dist_info(), init_dist(), _init_dist_pytorch(), _init_dist_slurm()

### Community 38 - "base_model.py"
Cohesion: 0.18
Nodes (3): CosineAnnealingRestartLR, get_position_from_periods(), MultiStepRestartLR

### Community 39 - "RetinaFace"
Cohesion: 0.17
Nodes (4): RetinaFace, decode(), decode_landm(), PriorBox

### Community 40 - "Model"
Cohesion: 0.13
Nodes (7): AutoShape, Concat, NMS, Model, check_anchor_order(), copy_attr(), fuse_conv_and_bn()

### Community 41 - "default_init_weights"
Cohesion: 0.13
Nodes (4): default_init_weights(), ResidualBlockNoBN, ResidualDenseBlock, RRDB

### Community 42 - "prefetch_dataloader.py"
Cohesion: 0.11
Nodes (3): CPUPrefetcher, PrefetchDataLoader, PrefetchGenerator

### Community 43 - "train_pipeline"
Cohesion: 0.18
Nodes (9): build_dataloader(), build_dataset(), worker_init_fn(), build_model(), create_train_val_dataloader(), train_pipeline(), check_resume(), make_exp_dirs() (+1 more)

### Community 44 - "psnr_ssim.py"
Cohesion: 0.22
Nodes (6): reorder_image(), to_y_channel(), calculate_psnr(), calculate_ssim(), _ssim(), bgr2ycbcr()

### Community 46 - "L1Loss"
Cohesion: 0.24
Nodes (3): l1_loss(), L1Loss, WeightedTVLoss

### Community 47 - "setup.py"
Cohesion: 0.07
Nodes (11): fused_leaky_relu(), FusedLeakyReLU, FusedLeakyReLUFunction, FusedLeakyReLUFunctionBackward, upfirdn2d(), UpFirDn2d, upfirdn2d_native(), UpFirDn2dBackward (+3 more)

### Community 49 - "train"
Cohesion: 0.14
Nodes (14): train, cross_entropy_loss, ema_decay, entropy_loss_weight, feat_loss_weight, fidelity_weight, manual_seed, net_d_iters (+6 more)

### Community 50 - "upfirdn2d_kernel.cu"
Cohesion: 0.26
Nodes (3): floor_div(), upfirdn2d_kernel(), upfirdn2d_kernel_large()

### Community 51 - "losses.py"
Cohesion: 0.17
Nodes (5): g_path_regularize(), gradient_penalty_loss(), mse_loss(), MSELoss, r1_penalty()

### Community 56 - "ResBlock"
Cohesion: 0.20
Nodes (3): Downsample, ResBlock, swish()

### Community 57 - "vgg_arch.py"
Cohesion: 0.15
Nodes (3): insert_bn(), VGGFeatureExtractor, PerceptualLoss

### Community 58 - ".__init__"
Cohesion: 0.20
Nodes (3): Encoder, GumbelQuantizer, VectorQuantizer

### Community 60 - "time"
Cohesion: 0.24
Nodes (4): get_time_str(), dict2str(), ordered_yaml(), parse()

### Community 61 - "detection/__init__.py"
Cohesion: 0.31
Nodes (5): init_detection_model(), init_retinaface_model(), init_yolov5face_model(), YoloDetector, load_file_from_url()

### Community 62 - ".batched_detect_faces"
Cohesion: 0.20
Nodes (3): batched_decode(), batched_decode_landm(), py_cpu_nms()

### Community 63 - "CodeFormer_colorization.yml"
Cohesion: 0.20
Nodes (9): datasets, dist_params, backend, port, find_unused_parameters, manual_seed, model_type, name (+1 more)

### Community 64 - "CodeFormer_inpainting.yml"
Cohesion: 0.20
Nodes (9): datasets, dist_params, backend, port, find_unused_parameters, manual_seed, model_type, name (+1 more)

### Community 65 - "CodeFormer_stage2.yml"
Cohesion: 0.20
Nodes (9): datasets, dist_params, backend, port, find_unused_parameters, manual_seed, model_type, name (+1 more)

### Community 66 - "CodeFormer_stage3.yml"
Cohesion: 0.20
Nodes (9): datasets, dist_params, backend, port, find_unused_parameters, manual_seed, model_type, name (+1 more)

### Community 67 - "VQGAN_512_ds32_nearest_stage1.yml"
Cohesion: 0.14
Nodes (13): datasets, dist_params, backend, port, find_unused_parameters, manual_seed, model_type, name (+5 more)

### Community 68 - "AttnBlock"
Cohesion: 0.22
Nodes (3): AttnBlock, Generator, Upsample

### Community 69 - "weighted_loss"
Cohesion: 0.31
Nodes (4): reduce_loss(), weight_reduce_loss(), weighted_loss(), wrapper()

### Community 76 - "network_g"
Cohesion: 0.22
Nodes (9): network_g, codebook_size, connect_list, dim_embd, fix_modules, n_head, n_layers, type (+1 more)

### Community 77 - "network_g"
Cohesion: 0.22
Nodes (9): network_g, codebook_size, connect_list, dim_embd, fix_modules, n_head, n_layers, type (+1 more)

### Community 78 - "network_g"
Cohesion: 0.22
Nodes (9): network_g, codebook_size, connect_list, dim_embd, fix_modules, n_head, n_layers, type (+1 more)

### Community 80 - "fused_bias_act.cpp"
Cohesion: 0.25
Nodes (3): fused_bias_act(), fused_bias_act_op(), PYBIND11_MODULE()

### Community 81 - "upfirdn2d.cpp"
Cohesion: 0.25
Nodes (3): upfirdn2d_op(), PYBIND11_MODULE(), upfirdn2d()

### Community 82 - "psnr"
Cohesion: 0.25
Nodes (8): psnr, crop_border, test_y_channel, type, val, metrics, save_img, val_freq

### Community 83 - "psnr"
Cohesion: 0.25
Nodes (8): psnr, crop_border, test_y_channel, type, val, metrics, save_img, val_freq

### Community 84 - "psnr"
Cohesion: 0.25
Nodes (8): psnr, crop_border, test_y_channel, type, val, metrics, save_img, val_freq

### Community 85 - "psnr"
Cohesion: 0.25
Nodes (8): psnr, crop_border, test_y_channel, type, val, metrics, save_img, val_freq

### Community 86 - "network_g"
Cohesion: 0.25
Nodes (8): network_g, codebook_size, connect_list, dim_embd, fix_modules, n_head, n_layers, type

### Community 87 - "psnr"
Cohesion: 0.25
Nodes (8): psnr, crop_border, test_y_channel, type, val, metrics, save_img, val_freq

### Community 90 - "Bottleneck"
Cohesion: 0.22
Nodes (3): Bottleneck, BottleneckCSP, C3

### Community 91 - "yolov5l.yaml"
Cohesion: 0.29
Nodes (6): anchors, backbone, depth_multiple, head, nc, width_multiple

### Community 92 - "yolov5n.yaml"
Cohesion: 0.29
Nodes (6): anchors, backbone, depth_multiple, head, nc, width_multiple

### Community 93 - "logger"
Cohesion: 0.29
Nodes (7): logger, print_freq, save_checkpoint_freq, use_tb_logger, wandb, project, resume_id

### Community 94 - "network_vqgan"
Cohesion: 0.29
Nodes (7): network_vqgan, ch_mult, codebook_size, img_size, nf, quantizer, type

### Community 95 - "path"
Cohesion: 0.29
Nodes (7): path, param_key_g, pretrain_network_d, pretrain_network_g, resume_state, strict_load_d, strict_load_g

### Community 96 - "logger"
Cohesion: 0.29
Nodes (7): logger, print_freq, save_checkpoint_freq, use_tb_logger, wandb, project, resume_id

### Community 97 - "network_vqgan"
Cohesion: 0.29
Nodes (7): network_vqgan, ch_mult, codebook_size, img_size, nf, quantizer, type

### Community 98 - "path"
Cohesion: 0.29
Nodes (7): path, param_key_g, pretrain_network_d, pretrain_network_g, resume_state, strict_load_d, strict_load_g

### Community 99 - "logger"
Cohesion: 0.29
Nodes (7): logger, print_freq, save_checkpoint_freq, use_tb_logger, wandb, project, resume_id

### Community 100 - "network_vqgan"
Cohesion: 0.29
Nodes (7): network_vqgan, ch_mult, codebook_size, img_size, nf, quantizer, type

### Community 101 - "path"
Cohesion: 0.29
Nodes (7): path, param_key_g, pretrain_network_d, pretrain_network_g, resume_state, strict_load_d, strict_load_g

### Community 102 - "logger"
Cohesion: 0.29
Nodes (7): logger, print_freq, save_checkpoint_freq, use_tb_logger, wandb, project, resume_id

### Community 103 - "network_vqgan"
Cohesion: 0.29
Nodes (7): network_vqgan, ch_mult, codebook_size, img_size, nf, quantizer, type

### Community 104 - "logger"
Cohesion: 0.29
Nodes (7): logger, print_freq, save_checkpoint_freq, use_tb_logger, wandb, project, resume_id

### Community 105 - "network_g"
Cohesion: 0.29
Nodes (7): network_g, ch_mult, codebook_size, img_size, nf, quantizer, type

### Community 106 - "path"
Cohesion: 0.29
Nodes (7): path, param_key_g, pretrain_network_d, pretrain_network_g, resume_state, strict_load_d, strict_load_g

### Community 112 - "network_d"
Cohesion: 0.33
Nodes (6): network_d, model_path, n_layers, nc, ndf, type

### Community 113 - "path"
Cohesion: 0.33
Nodes (6): path, param_key_g, pretrain_network_d, pretrain_network_g, resume_state, strict_load_g

### Community 116 - "master_only"
Cohesion: 0.19
Nodes (7): init_loggers(), master_only(), wrapper(), get_env_info(), init_tb_logger(), init_wandb_logger(), MessageLogger

### Community 117 - "optim_g"
Cohesion: 0.40
Nodes (5): betas, lr, type, weight_decay, optim_g

### Community 118 - "network_d"
Cohesion: 0.40
Nodes (5): network_d, n_layers, nc, ndf, type

### Community 121 - "scheduler"
Cohesion: 0.50
Nodes (4): gamma, milestones, type, scheduler

### Community 122 - "optim_g"
Cohesion: 0.40
Nodes (5): betas, lr, type, weight_decay, optim_g

### Community 123 - "optim_d"
Cohesion: 0.40
Nodes (5): betas, lr, type, weight_decay, optim_d

### Community 124 - "optim_g"
Cohesion: 0.40
Nodes (5): betas, lr, type, weight_decay, optim_g

### Community 127 - "perceptual_opt"
Cohesion: 0.40
Nodes (5): loss_weight, range_norm, type, use_input_norm, perceptual_opt

### Community 132 - "scheduler"
Cohesion: 0.40
Nodes (5): eta_min, periods, restart_weights, type, scheduler

### Community 134 - "scheduler"
Cohesion: 0.50
Nodes (4): gamma, milestones, type, scheduler

### Community 135 - "pixel_opt"
Cohesion: 0.50
Nodes (4): loss_weight, reduction, type, pixel_opt

## Knowledge Gaps
- **18 isolated node(s):** `up_x`, `up_y`, `down_x`, `down_y`, `pad_x0` (+13 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 984 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **40 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_root_logger()` connect `get_root_logger` to `data_util.py`, `deform_conv.py`, `CodeFormerIdxModel`, `torch`, `dist_util.py`, `.model_ema`, `CodeFormerJointModel`, `.__getitem__`, `CodeFormerModel`, `train_pipeline`, `SRModel`, `ffhq_blind_joint_dataset.py`, `master_only`, `arch_util.py`, `VQGANModel`, `.__init__`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **What connects `up_x`, `up_y`, `down_x` to the rest of the system?**
  _18 weakly-connected nodes found - possible missing edges._
- **Should `data_util.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12121212121212122 - nodes in this community are weakly interconnected._
- **Why does `BaseModel` connect `BaseModel` to `torch`, `.model_ema`, `.save_network`, `SRModel`, `get_root_logger`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Should `check_evidence.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12222222222222222 - nodes in this community are weakly interconnected._
- **Why does `tensor2img()` connect `torch` to `CodeFormerIdxModel`, `numpy`, `CodeFormerJointModel`, `CodeFormerModel`, `SRModel`, `cv2`, `facelib/utils/misc.py`, `VQGANModel`, `matlab_functions.py`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Should `bisenet.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05725490196078432 - nodes in this community are weakly interconnected._