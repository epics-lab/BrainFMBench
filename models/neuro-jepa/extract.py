"""Neuro-JEPA-B for BrainFMBench: mean of the final-layer patch tokens, 768 dimensions.

Model code comes from https://github.com/NYUMedML/Neuro-JEPA at commit 2df20d3,
cloned on the cluster at NEUROJEPA_SRC (the repo's src/ folder). Weights and config
come from NYUMedML/Neuro-JEPA on Hugging Face via weights.txt.

Input preparation follows notebooks/extract_feature_example.ipynb in that repo,
after turning the turboprep-affine z-scored volume back into a skull-stripped image
(brain > 0, background = 0), which is what Neuro-JEPA saw in pretraining.

The checkpoint holds only the backbone (no pretrained attentive pooler) and the
benchmark requires frozen features, so tokens are mean-pooled.
Runs in the shared evaluation environment: no torchmetrics, hydra, timm or
safetensors needed.
"""
import os
import sys
import glob
import json
import struct
import types
from functools import partial

import numpy as np
import torch
import torch.nn as nn

NEUROJEPA_SRC = os.environ.get("NEUROJEPA_SRC", "/home/arelbaha/links/scratch/neurojepa_git/src")
EMBED_DIM = 768
SEED = 0


def _import_vit():
    sys.path.insert(0, NEUROJEPA_SRC)
    try:
        import timm.layers  # noqa: F401
    except ImportError:
        # Only drop_path is used from timm, and it is the identity at inference.
        stub = types.ModuleType("timm.layers")
        stub.drop_path = lambda x, drop_prob=0.0, training=False, scale_by_keep=True: x
        sys.modules.setdefault("timm", types.ModuleType("timm"))
        sys.modules["timm.layers"] = stub
    import neurojepa.models.vision_transformer as vit
    return vit


def _load_safetensors(path):
    try:
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    except ImportError:
        pass
    dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        buf = bytearray(f.read())
    header.pop("__metadata__", None)
    out = {}
    for k, v in header.items():
        a, b = v["data_offsets"]
        t = torch.frombuffer(buf, dtype=dtypes[v["dtype"]], count=(b - a) // dtypes[v["dtype"]].itemsize, offset=a)
        out[k] = t.reshape(v["shape"]).clone()
    return out


def _ns(v):
    if isinstance(v, dict):
        return types.SimpleNamespace(**{k: _ns(x) for k, x in v.items()})
    return [_ns(x) for x in v] if isinstance(v, list) else v


def load_model(weights_dir, device):
    vit = _import_vit()
    cfg = json.load(open(os.path.join(weights_dir, "config.json")))
    m, meta = cfg["model"], cfg.get("meta", {})
    model = vit.__dict__[m["backbone_name"]](
        img_size=m["img_size"], patch_size=m["patch_size"], in_chans=m.get("in_chans", 1),
        out_layers=None, uniform_power=m.get("uniform_power", False),
        use_sdpa=meta.get("use_sdpa", False), use_silu=m.get("use_silu", False),
        wide_silu=m.get("wide_silu", False), use_rope=m.get("use_rope", False),
        use_activation_checkpointing=False,
        use_moe=m.get("use_moe", False), moe_params=_ns(m.get("moe_params")),
    )
    sd = _load_safetensors(os.path.join(weights_dir, "model.safetensors"))
    sd = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def _transforms():
    from monai import transforms as T

    def nan0(x):
        x[torch.isnan(x)] = 0.0
        return x

    def apply_mask(d):
        # turboprep-affine is z-scored with background near -60: shift the brain to > 0
        # and zero everything outside mask.nii.gz.
        img, msk = d["image"], d["mask"] > 0
        img = img - img[msk].min() + 1e-3
        img[~msk] = 0.0
        d["image"] = img
        return d

    return T.Compose([
        T.LoadImaged(keys=["image", "mask"]),
        T.EnsureChannelFirstd(keys=["image", "mask"]),
        T.Lambdad(("image",), nan0),
        T.Lambda(apply_mask),
        T.Orientationd(keys=["image"], axcodes="RAS"),
        T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=[5]),
        T.CropForegroundd(keys=["image"], source_key="image",
                          select_fn=lambda x: x > 0.0, margin=4, allow_smaller=True),
        T.ResizeWithPadOrCropd(keys=["image"], spatial_size=[180, 216, 180], mode="edge"),
        T.Resized(keys=["image"], spatial_size=[100, 120, 100]),
        T.CenterSpatialCropd(keys=["image"], roi_size=[96, 108, 96]),
        T.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5,
                                          b_min=0, b_max=1, clip=True),
        T.CastToTyped(keys=["image"], dtype=np.float32),
    ])


def extract(input_dir, output_csv, weights_dir):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(weights_dir, device)
    trans = _transforms()
    paths = sorted(glob.glob(os.path.join(input_dir, "*", "normalized.nii.gz")))
    print(f"[neuro-jepa] {len(paths)} subjects in {input_dir} on {device}", flush=True)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w") as f, torch.no_grad():
        f.write("subject_id," + ",".join(f"f{i}" for i in range(EMBED_DIM)) + "\n")
        for i, p in enumerate(paths):
            d = trans({"image": p, "mask": os.path.join(os.path.dirname(p), "mask.nii.gz")})
            x = torch.as_tensor(d["image"]).as_subclass(torch.Tensor)[None].to(device)
            tokens, _ = model(x)                                # [1, 576, 768]
            feat = tokens.float().mean(dim=1)[0].cpu().numpy()
            f.write(os.path.basename(os.path.dirname(p)) + "," +
                    ",".join(f"{v:.7g}" for v in feat) + "\n")
            if (i + 1) % 50 == 0:
                print(f"[neuro-jepa] {i + 1}/{len(paths)}", flush=True)
    print(f"[neuro-jepa] wrote {len(paths)} x {EMBED_DIM} -> {output_csv}", flush=True)
