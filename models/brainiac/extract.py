"""BrainIAC for BrainFMBench: ViT backbone, CLS token, 768 dimensions.

Input preparation and readout follow the extraction used for the MedAGI 2026 paper:
resize to 96^3 (trilinear), z-score over the whole volume, first output token.
Checkpoint: BrainIAC.ckpt from https://github.com/AIM-KannLab/BrainIAC
"""
import os
import glob

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH = 16

def _setup():
    # Same settings as the MedAGI 2026 extraction: deterministic, single threaded, CPU
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _write(output_csv, sids, feats, tag):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w") as f:
        f.write("subject_id," + ",".join(f"f{i}" for i in range(feats.shape[1])) + "\n")
        for sid, row in zip(sids, feats):
            f.write(sid + "," + ",".join(str(np.float32(v)) for v in row) + "\n")
    print(f"[{tag}] wrote {feats.shape[0]} x {feats.shape[1]} -> {output_csv}", flush=True)


def _subjects(input_dir, tag):
    paths = sorted(glob.glob(os.path.join(input_dir, "*", "normalized.nii.gz")))
    sids = [os.path.basename(os.path.dirname(p)) for p in paths]
    print(f"[{tag}] {len(paths)} subjects in {input_dir}", flush=True)
    return paths, sids


def _prepare(path):
    img = nib.load(path).get_fdata().astype(np.float32)
    if img.shape != (96, 96, 96):
        img = F.interpolate(torch.from_numpy(img[None, None]), size=(96, 96, 96),
                            mode="trilinear", align_corners=False).squeeze().numpy()
    img = torch.from_numpy(img)
    return ((img - img.mean()) / (img.std() + 1e-6))[None]


def _load(ckpt_path):
    from monai.networks.nets import ViT
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    vit = ViT(in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
              hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12, save_attn=True)
    pe_key = "backbone.patch_embedding.position_embeddings"
    if pe_key in sd:
        vit.patch_embedding.position_embeddings = nn.Parameter(sd[pe_key].clone())
    vit.load_state_dict({k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")},
                        strict=False)
    return vit.eval()


def extract(input_dir, output_csv, weights_dir):
    _setup()
    paths, sids = _subjects(input_dir, "brainiac")
    vit = _load(os.path.join(weights_dir, "BrainIAC.ckpt"))
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            out = vit(torch.stack([_prepare(p) for p in paths[i:i + BATCH]]))
            feats.append((out[0][:, 0] if isinstance(out, tuple) else out[:, 0]).numpy())
            print(f"[brainiac] {min(i + BATCH, len(paths))}/{len(paths)}", flush=True)
    _write(output_csv, sids, np.vstack(feats), "brainiac")
