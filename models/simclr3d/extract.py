"""3D-Neuro-SimCLR for BrainFMBench: ResNet-18 encoder, projector removed, 512 dimensions.

Input preparation follows the reference preprocessing in the 3D-Neuro-SimCLR repo
(datasets/t1_datasets.py): transpose (x, y, z) -> (z, y, x), fixed crop [10:160, 19:211, 1:]
to 150 x 192 x 192, standardization within the TurboPrep brain mask with background set to 0.
Requires mask.nii.gz next to normalized.nii.gz.
Checkpoint: simclr_3d_brain_foundation.tar from https://github.com/emilykaczmarek/3D-Neuro-SimCLR
"""
import os
import glob

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn

BATCH = 8

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
    msk = nib.load(os.path.join(os.path.dirname(path), "mask.nii.gz")).get_fdata().astype(np.float32)
    if img.shape != msk.shape:
        raise ValueError(f"Shape mismatch: {path}: {img.shape} vs {msk.shape}")
    img = np.transpose(img, (2, 1, 0))[10:160, 19:211, 1:]
    msk = np.transpose(msk, (2, 1, 0))[10:160, 19:211, 1:]
    img = img * msk
    n = max(float(msk.sum()), 1.0)
    mean = img.sum() / n
    std = np.sqrt((img ** 2).sum() / n - mean ** 2 + 1e-12)
    img = (img - mean) / std * msk
    return torch.from_numpy(img[None]).float()


def _load(ckpt_path):
    from monai.networks.nets import resnet18
    enc = resnet18(spatial_dims=3, n_input_channels=1, num_classes=0)
    enc.fc = nn.Identity()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc.load_state_dict({k.replace("module.encoder.", ""): v for k, v in ckpt["model_state_dict"].items()
                         if "encoder" in k and "projector" not in k}, strict=True)
    return enc.eval()


def extract(input_dir, output_csv, weights_dir):
    _setup()
    paths, sids = _subjects(input_dir, "simclr3d")
    enc = _load(os.path.join(weights_dir, "simclr_3d_brain_foundation.tar"))
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            feats.append(enc(torch.stack([_prepare(p) for p in paths[i:i + BATCH]])).numpy())
            print(f"[simclr3d] {min(i + BATCH, len(paths))}/{len(paths)}", flush=True)
    _write(output_csv, sids, np.vstack(feats), "simclr3d")
