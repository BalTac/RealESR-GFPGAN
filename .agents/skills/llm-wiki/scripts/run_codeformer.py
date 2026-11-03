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

# Add CodeFormer repo to path
codeformer_dir = os.path.abspath('CodeFormer_src/CodeFormer-master')
if codeformer_dir not in sys.path:
    sys.path.insert(0, codeformer_dir)

from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.misc import get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY

# Ensure architectures are registered
import basicsr.archs
try:
    import CodeFormer_src.CodeFormer_master.basicsr.archs
except Exception:
    pass

def run():
    device = get_device()
    print(f"Device: {device}")

    # Set weights paths
    weights_dir = os.path.join(codeformer_dir, 'weights')
    codeformer_weights = os.path.join(weights_dir, 'CodeFormer', 'codeformer.pth')
    
    # Initialize CodeFormer Net
    print("Loading CodeFormer architecture...")
    net = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=['32', '64', '128', '256']
    ).to(device)

    print(f"Loading checkpoint from {codeformer_weights}...")
    checkpoint = torch.load(codeformer_weights, map_location=device)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()
    print("CodeFormer ready!")

    os.makedirs('results/codeformer', exist_ok=True)

    test_images = [
        ('Foto_3', 'inputs/whole_imgs/Foto_3.jpg'),
        ('Blake_Lively', 'inputs/whole_imgs/Blake_Lively.jpg'),
        ('10045', 'inputs/whole_imgs/10045.png'),
        ('00', 'inputs/whole_imgs/00.jpg')
    ]

    # Fidelity weight: w=0.6 provides excellent balance of texture and restoration
    w = 0.6
    upscale = 2

    # Ensure weights/facelib exists in working directory
    os.makedirs('weights/facelib', exist_ok=True)
    import shutil
    if not os.path.exists('weights/facelib/detection_Resnet50_Final.pth'):
        shutil.copy('gfpgan/weights/detection_Resnet50_Final.pth', 'weights/facelib/detection_Resnet50_Final.pth')
    if not os.path.exists('weights/facelib/parsing_parsenet.pth'):
        shutil.copy('gfpgan/weights/parsing_parsenet.pth', 'weights/facelib/parsing_parsenet.pth')

    # Initialize FaceRestoreHelper with the already downloaded weights
    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device
    )

    for name, img_path in test_images:
        ext = os.path.splitext(img_path)[1]
        print(f"\n--- Processing {name} ({img_path}) ---")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error loading {img_path}")
            continue

        start_t = time.time()
        face_helper.clean_all()
        face_helper.read_image(img)
        num_faces = face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        print(f"  Detected faces: {num_faces}")
        face_helper.align_warp_face()

        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            with torch.no_grad():
                output = net(cropped_face_t, w=w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)

        # Paste back into upsampled base image
        face_helper.get_inverse_affine(None)
        restored_img = face_helper.paste_faces_to_input_image()

        elapsed = time.time() - start_t
        out_path = f"results/codeformer/codeformer_{name}{ext}"
        cv2.imwrite(out_path, restored_img)
        print(f"  Saved: {out_path} ({restored_img.shape[1]}x{restored_img.shape[0]}) in {elapsed:.2f}s")

if __name__ == '__main__':
    run()
