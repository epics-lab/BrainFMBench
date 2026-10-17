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
PATCH = 16
DO_RESAMPLE = False

def resample_1mm(t, zooms):
    # Matches monai Spacingd(pixdim=1, mode="bilinear", padding_mode="border"):
    in_shape = t.shape[2:]
    out_shape = [int(np.floor((n - 1) * z + 1e-6)) + 1 for n, z in zip(in_shape, zooms)]
    axes = [torch.arange(m, dtype=torch.float64) / z / (n - 1) * 2 - 1
            for m, n, z in zip(out_shape, in_shape, zooms)]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack([gz, gy, gx], dim=-1)[None].float()  # grid_sample wants (W, H, D) order
    return F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)


def load_volume(path):
    img = nib.load(path)
    v = img.get_fdata(dtype=np.float32)

    # SignalFillEmptyd equiv
    v = np.nan_to_num(v, nan=0.0)

    # NormalizeIntensityd equiv
    for s in v:
        m = s != 0
        if m.any():
            x = s[m]
            sd = x.std()
            s[m] = (x - x.mean()) / (sd if sd != 0 else 1.0)

    t = torch.from_numpy(v)[None, None]

    if DO_RESAMPLE:
        zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
        if not np.allclose(zooms, 1.0):
            t = resample_1mm(t, zooms)

    # Pad with 0 now after z-score
    # This is done symmetrically like monai
    pads = []
    for n in reversed(t.shape[2:]):
        p = (-n) % PATCH
        pads += [p // 2, p - p // 2]
    return F.pad(t, pads, value=0.0)



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
