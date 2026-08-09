# Contributing a model to BrainFMBench

BrainFMBench evaluates **frozen features** from brain MRI models on
neuroimaging datasets (NKI, HBN) across sex classification and age / BMI
regression. You submit your model; we run feature extraction on our cluster
against preprocessed data (turboprep), then score the returned features.

## What you submit

Open a Pull Request adding one folder under `models/`:

```
models/<your-model>/
  model.yaml        # metadata
  extract.py        # your feature-extraction code (fixed interface, see below)
  weights.txt       # direct-download URL(s) to your checkpoint(s)
  requirements.txt  # optional; pinned dependencies for your extract.py
```

You do **not** upload weights to this repo. Host them yourself and list the
URL(s); we download them at run time.

See `example-submission/` for a minimal, working template you can copy.

## `extract.py` — the interface

Your `extract.py` must define exactly this function:

```python
def extract(input_dir, output_csv, weights_dir):
    ...
```

- **`input_dir`** — a directory of preprocessed volumes, one folder per subject:
  `<input_dir>/<subject_id>/normalized.nii.gz`. (This is turboprep output;
  `cat12` is also available if you set `preprocessing: cat12`.)
- **`weights_dir`** — a directory containing the checkpoint file(s) downloaded
  from your `weights.txt`.
- **`output_csv`** — write features here as `subject_id,f0,f1,...,fN`,
  one row per subject.

Extract **frozen features only** (a forward pass / embedding). Do not train.
Determinism is expected — set your seeds.

## `model.yaml`

```yaml
name: My Model
authors:
  - Lastname, F., et al.
submitted_by: Your Name
description: one-line description
preprocessing: turboprep        # turboprep (default) or cat12
embedding_dim: 768              # your feature dimension
architecture: short description
datasets: [NKI, HBN]
tasks: [sex, age, bmi]
```

## `weights.txt`

One direct-download URL per line. Supported hosts:

- HuggingFace — use the `/resolve/` URL, e.g.
  `https://huggingface.co/<user>/<model>/resolve/main/checkpoint.pth`
- Zenodo, GitHub Releases, or an institutional raw link (`/-/raw/`)

Not supported: Google Drive, Dropbox, or `/blob/` viewer pages (they return
HTML, not the file). Multi-file checkpoints: one URL per line.

## The evaluation environment

By default your code runs in a shared PyTorch environment on the cluster.
Available packages include: **torch, torchvision, monai, nibabel, numpy, scipy,
scikit-learn, pandas, nilearn, einops, SimpleITK, tqdm** and more.

If your model needs versions that differ from the shared environment, add a
`requirements.txt` to your submission folder. When present, extraction runs in
an isolated virtualenv built from that file instead of the shared environment.
Pin exact versions:

```
torch==2.4.0
monai==1.3.0
```

Every package must be available in the Alliance wheelhouse — cluster compute
nodes have no outbound network, so PyPI is unreachable at build time. Check
availability at https://docs.alliancecan.ca/wiki/Available_Python_wheels before
submitting. If something you need is missing, open an issue and we will work
out an alternative. The venv is built once per submission, reused across
datasets, and rebuilt only when `requirements.txt` changes.

## What happens after you open the PR

1. Automated validation checks your submission is well-formed (yaml parses,
   `extract.py` defines `extract(...)`, weights URLs resolve). If you supplied
   a `requirements.txt`, your `extract.py` is run against a synthetic volume
   inside an environment built from it.
2. A maintainer reviews and merges.
3. On merge, extraction runs on the cluster; the resulting features are scored
   and the leaderboard updates automatically.

Because extraction runs on our allocation, only maintainer-approved submissions
are executed — we never run unreviewed code automatically.
