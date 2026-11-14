"""Dinov3D-neuro for BrainFMBench.

Model code comes from asagilmore/dinov3d (private), cloned on Rorqual at
DINOV3D_SRC and pinned to commit e5ef877. Weights and config come from
huggingbrain/Dinov3d-Neuro via weights.txt.
"""
import os
import sys
import glob

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

DINOV3D_SRC = "/home/arelbaha/links/scratch/dinov3d_git"
CROP_MARGIN = 5     # voxels of background kept around the brain bounding box
TARGET_SIZE = 256   # final cube side fed to the model (multiple of the 16-voxel patch)
EPS = 1e-6


def load_volume(path, margin=CROP_MARGIN, size=TARGET_SIZE):
    img = nib.load(path)
    assert np.allclose(img.header.get_zooms()[:3], 1.0, atol=1e-3), f"{path} is not 1mm isotropic"
    v = np.nan_to_num(img.get_fdata().astype(np.float32), nan=0.0)

    # Z-score over brain voxels only; background becomes 0.
    brain = v > v.min() + EPS
    assert brain.any(), f"{path} has no voxels above background"
    x = v[brain]
    out = np.zeros_like(v)
    out[brain] = (x - x.mean()) / (x.std() + EPS)

    # Crop to the brain's axis-aligned bounding box plus `margin` voxels per side.
    # Zero-pad first so the margin is exact even when the brain touches the edge.
    out = np.pad(out, margin)
    idx = np.nonzero(np.pad(brain, margin))
    out = out[tuple(slice(i.min() - margin, i.max() + margin + 1) for i in idx)]

    # Pad (centered, with background) to a cube so the resize is isotropic.
    side = max(out.shape)
    out = np.pad(out, [((side - n) // 2, side - n - (side - n) // 2) for n in out.shape])

    # Isotropic resample to size^3.
    t = torch.from_numpy(out)[None, None]
    return F.interpolate(t, size=(size,) * 3, mode="trilinear", align_corners=False)


def extract(input_dir, output_csv, weights_dir):
    sys.path.insert(0, DINOV3D_SRC)
    from dinov3.inference.load import load_teacher_model_for_inference

    if os.environ.get("SLURM_CPUS_PER_TASK"):
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.manual_seed(0)

    model, dim = load_teacher_model_for_inference(
        os.path.join(weights_dir, "config.yaml"),
        os.path.join(weights_dir, "teacher_checkpoint.pth"),
        device="cpu",
    )
    model.eval()

    paths = sorted(glob.glob(os.path.join(input_dir, "*", "normalized.nii.gz")))
    print(f"[dinov3d] {len(paths)} subjects in {input_dir}, embed_dim={dim}", flush=True)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w") as f, torch.no_grad():
        f.write("subject_id," + ",".join(f"f{i}" for i in range(dim)) + "\n")
        for i, p in enumerate(paths):
            cls = model.forward_features(load_volume(p))["x_norm_clstoken"][0].float().numpy()
            f.write(os.path.basename(os.path.dirname(p)) + "," + ",".join(f"{v:.7g}" for v in cls) + "\n")
            if (i + 1) % 50 == 0:
                print(f"[dinov3d] {i + 1}/{len(paths)}", flush=True)
    print(f"[dinov3d] wrote {len(paths)} x {dim} -> {output_csv}", flush=True)
