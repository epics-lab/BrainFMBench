"""
Example BrainFMBench submission.

Every submission must define ONE function with this exact signature:

    extract(input_dir, output_csv, weights_dir)

  input_dir    : directory of preprocessed volumes, one folder per subject:
                   <input_dir>/<subject_id>/normalized.nii.gz
                 (this is OpenMRIdatasets/<DATASET>/turboprep/ on the cluster)
  output_csv   : path to write features to, as:
                   subject_id,f0,f1,...,fN     (one row per subject)
  weights_dir  : directory holding your downloaded checkpoint file(s)
                 (whatever URLs you list in weights.txt land here)

Your code runs on the cluster in an isolated virtualenv built from your
requirements.txt, if you provide one; otherwise in the shared evaluation
environment (PyTorch + MONAI + nibabel + numpy + ...). Extract frozen
features only: load each subject's volume, run your model forward, write
the embedding.

Replace the body of `embed_one` with your model's forward pass.
"""
import os
import glob
import csv

import numpy as np
import nibabel as nib


EMBED_DIM = 32  # this example writes 32 features; set to your model's dim


def load_model(weights_dir):
    """Load your model from the checkpoint(s) in weights_dir.
    This example has no real weights, so it returns None."""
    # e.g.:  ckpt = os.path.join(weights_dir, "my_checkpoint.pth")
    #        model = MyNet(); model.load_state_dict(torch.load(ckpt)); model.eval()
    #        return model
    return None


def embed_one(volume, model):
    """Return a 1-D feature vector for one subject's volume.
    Replace this with your model's forward pass.
    This example: a normalized 32-bin intensity histogram (deterministic)."""
    data = volume.astype(np.float32)
    data = data[data > 0]                      # ignore background
    hist, _ = np.histogram(data, bins=EMBED_DIM, range=(0, np.percentile(data, 99)))
    hist = hist / (hist.sum() + 1e-8)
    return hist.astype(np.float32)


def extract(input_dir, output_csv, weights_dir):
    model = load_model(weights_dir)

    subject_dirs = sorted(glob.glob(os.path.join(input_dir, "*", "normalized.nii.gz")))
    print(f"[example] found {len(subject_dirs)} subjects in {input_dir}")

    rows = []
    for vol_path in subject_dirs:
        subject_id = os.path.basename(os.path.dirname(vol_path))
        vol = nib.load(vol_path).get_fdata()
        feats = embed_one(vol, model)
        rows.append((subject_id, feats))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id"] + [f"f{i}" for i in range(EMBED_DIM)])
        for subject_id, feats in rows:
            w.writerow([subject_id] + [f"{v:.6f}" for v in feats])
    print(f"[example] wrote {len(rows)} rows x {EMBED_DIM} features -> {output_csv}")
