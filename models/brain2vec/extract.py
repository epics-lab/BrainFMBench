"""Brain2Vec feature extraction: CLS token of the encoder, 768 dims.

Weights are gated and staged on the cluster by the maintainers; they are read from
weights_dir and never downloaded at runtime (compute nodes are offline).
"""
import os
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from Brain2Vec.dataloader.loader import NiftiLoader
from Brain2Vec.models.brain2vec import brain2vec_base_patch8_dec512d8b


def extract(input_dir, output_csv, weights_dir):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dir = Path(input_dir) / "brain2vec_aligned_scans"
    if not in_dir.is_dir():
        in_dir = Path(input_dir)

    model = brain2vec_base_patch8_dec512d8b(img_size=128)
    model.load_state_dict(load_file(os.path.join(weights_dir, "brain2vec.safetensors"), device=device))
    model.to(device).eval()

    workers = max(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1, 0)
    loader = DataLoader(NiftiLoader(mode="inference", in_dir=in_dir, augmentation=False),
                        batch_size=1, shuffle=False, num_workers=workers)

    rows = []
    with torch.no_grad():
        for imgs, mrids in loader:
            imgs = imgs.to(device).float()
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                feats, _, _ = model.forward_encoder(imgs, mask_ratio=0.0)
            for name, vec in zip(mrids, feats[:, 0, :].float().cpu()):
                sid = name[:-7] if name.endswith(".nii.gz") else name
                rows.append((sid, vec.tolist()))
    rows.sort()

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id"] + [f"f{i}" for i in range(len(rows[0][1]))])
        for sid, vec in rows:
            w.writerow([sid] + [f"{v:.7g}" for v in vec])
    print(f"[brain2vec] wrote {len(rows)} x {len(rows[0][1])} -> {output_csv}", flush=True)
