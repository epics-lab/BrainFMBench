"""SwinBrain for BrainFMBench: SwinUNETR encoder10 output, average pooled, 384 dimensions.

Input preparation and readout follow the extraction used for the MedAGI 2026 paper:
resize to 128 x 128 x 64 (trilinear), z-score over the whole volume, repeated to 3 channels.
Checkpoint: Brain_Swin_UNETR.pth from https://github.com/MAI-Lab-West-China-Hospital/SwinBrain
"""
import os
import glob

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    img = torch.from_numpy(nib.load(path).get_fdata().astype(np.float32)[None, None])
    img = F.interpolate(img, size=(128, 128, 64), mode="trilinear", align_corners=False).squeeze(0)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return img.repeat(3, 1, 1, 1)


def _load(ckpt_path):
    from monai.networks.nets import SwinUNETR
    model = SwinUNETR(img_size=(128, 128, 64), in_channels=3, out_channels=3, spatial_dims=3,
                      feature_size=24, drop_rate=0.0, attn_drop_rate=0.0)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
    return model.eval()


def extract(input_dir, output_csv, weights_dir):
    _setup()
    paths, sids = _subjects(input_dir, "swinbrain")
    model = _load(os.path.join(weights_dir, "Brain_Swin_UNETR.pth"))
    hook = {}
    model.encoder10.register_forward_hook(lambda m, i, o: hook.__setitem__("x", o))
    pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            model(torch.stack([_prepare(p) for p in paths[i:i + BATCH]]))
            pooled = pool(hook["x"])
            feats.append(pooled.view(pooled.size(0), -1).numpy())
            print(f"[swinbrain] {min(i + BATCH, len(paths))}/{len(paths)}", flush=True)
    _write(output_csv, sids, np.vstack(feats), "swinbrain")
