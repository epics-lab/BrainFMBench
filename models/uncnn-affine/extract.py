"""un-CNN for BrainFMBench, inlined from epics-lab/untrained-cnn uncnn.py with config="robust"."""
import os
import glob

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
BATCH = 4


def load_and_resize(path):
    img = nib.load(path).get_fdata().astype(np.float32)
    if img.shape != (91, 109, 91):
        img = ndi.zoom(img, [91 / img.shape[0], 109 / img.shape[1], 91 / img.shape[2]], order=1)
    return img


def norm(ch):
    mn, mx = ch.min(), ch.max()
    if mx - mn < 1e-8:
        return np.zeros_like(ch)
    return (ch - mn) / (mx - mn)


def sobel_mag(img):
    return np.sqrt(ndi.sobel(img, axis=0) ** 2 + ndi.sobel(img, axis=1) ** 2 + ndi.sobel(img, axis=2) ** 2)


def rank_filter(img, size=3):
    return norm(ndi.percentile_filter(img, percentile=50, size=size))


def scale_robust(img, clip_low=2, clip_high=98, eps=1e-8):
    mask = img > np.percentile(img, 5)
    lo, hi = np.percentile(img[mask], [clip_low, clip_high])
    clipped = np.clip(img, lo, hi)
    brain = clipped[mask]
    q25, q75 = np.percentile(brain, [25, 75])
    out = np.zeros_like(clipped, dtype=np.float32)
    out[mask] = (brain - float(np.median(brain))) / (float(q75 - q25) + eps)
    return out


def to_input(path):
    img = scale_robust(load_and_resize(path))
    return np.stack([img, rank_filter(img), norm(sobel_mag(img))], axis=0).astype(np.float32)


class DepthwiseSepConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv3d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class UnCNN(nn.Module):
    def __init__(self, in_channels=3, seed=SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1a = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.norm1a = nn.GroupNorm(8, 64)
        self.conv1b = nn.Conv3d(64, 64, 3, padding=1)
        self.norm1b = nn.GroupNorm(8, 64)
        self.down1 = nn.AvgPool3d(2)
        self.conv2a = DepthwiseSepConv3d(64, 128)
        self.norm2a = nn.GroupNorm(8, 128)
        self.conv2b = DepthwiseSepConv3d(128, 128)
        self.norm2b = nn.GroupNorm(8, 128)
        self.down2 = nn.AvgPool3d(2)
        self.conv3a = DepthwiseSepConv3d(128, 256)
        self.norm3a = nn.GroupNorm(8, 256)
        self.conv3b = DepthwiseSepConv3d(256, 256)
        self.norm3b = nn.GroupNorm(8, 256)
        self.down3 = nn.AvgPool3d(2)
        self.conv4a = DepthwiseSepConv3d(256, 512)
        self.norm4a = nn.GroupNorm(8, 512)
        self.conv4b = DepthwiseSepConv3d(512, 512)
        self.norm4b = nn.GroupNorm(8, 512)
        self.down4 = nn.AvgPool3d(2)
        self.avg_pool = nn.AdaptiveAvgPool3d(2)

    def _cov_pool(self, x, max_ch=32):
        b, c = x.size(0), min(x.size(1), max_ch)
        mean_feats = self.avg_pool(x).view(b, -1)
        flat = x[:, :c].reshape(b, c, -1)
        flat = flat - flat.mean(dim=2, keepdim=True)
        cov = torch.bmm(flat, flat.transpose(1, 2)) / (flat.size(2) - 1)
        idx = torch.triu_indices(c, c, offset=1)
        return torch.cat([mean_feats, cov[:, idx[0], idx[1]]], dim=1)

    def forward(self, x):
        x1 = F.relu(self.norm1a(self.conv1a(x)))
        x1 = self.down1(F.relu(self.norm1b(self.conv1b(x1))))
        x2 = F.relu(self.norm2a(self.conv2a(x1)))
        x2 = self.down2(F.relu(self.norm2b(self.conv2b(x2))))
        x3 = F.relu(self.norm3a(self.conv3a(x2)))
        x3 = self.down3(F.relu(self.norm3b(self.conv3b(x3))))
        x4 = F.relu(self.norm4a(self.conv4a(x3)))
        x4 = self.down4(F.relu(self.norm4b(self.conv4b(x4))))
        return torch.cat([self._cov_pool(x1), self._cov_pool(x2), self._cov_pool(x3), self._cov_pool(x4)], dim=1)


def extract(input_dir, output_csv, weights_dir):
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))

    paths = sorted(glob.glob(os.path.join(input_dir, "*", "normalized.nii.gz")))
    sids = [os.path.basename(os.path.dirname(p)) for p in paths]
    print(f"[uncnn] {len(paths)} subjects in {input_dir}", flush=True)

    model = UnCNN().eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH):
            x = torch.from_numpy(np.stack([to_input(p) for p in paths[i:i + BATCH]]))
            feats.append(model(x).numpy())
            print(f"[uncnn] {min(i + BATCH, len(paths))}/{len(paths)}", flush=True)
    feats = np.vstack(feats)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w") as f:
        f.write("subject_id," + ",".join(f"f{i}" for i in range(feats.shape[1])) + "\n")
        for sid, row in zip(sids, feats):
            f.write(sid + "," + ",".join(f"{v:.7g}" for v in row) + "\n")
    print(f"[uncnn] wrote {feats.shape[0]} x {feats.shape[1]} -> {output_csv}", flush=True)
