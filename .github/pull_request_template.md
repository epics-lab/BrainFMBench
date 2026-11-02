<!--
Thanks for submitting a model to BrainFMBench!
Please fill in the sections below and check the boxes that apply.
See CONTRIBUTING.md for the full guide.
-->

## Model

- **Name:**
- **Architecture / backbone:**
- **Embedding dimension:**
- **Paper / preprint link:**
- **Model card (e.g. Hugging Face):**
- **Code repository:**
- **Was the model designed or tuned on NKI or HBN?** <!-- yes/no; if yes, explain -->

## What this PR adds

<!-- e.g. "Adds models/my-model/ with extract.py, model.yaml, and weights.txt" -->

## Submission type

- [ ] **Code submission** — includes `extract.py` + `weights.txt`; features are extracted on the cluster
- [ ] **Features-only submission** — includes a precomputed `features/<DATASET>.csv` I generated myself

---

## Checklist

### Files
- [ ] Everything is under `models/<my-model>/`
- [ ] `model.yaml` includes `name`, `authors`, `preprocessing`, `embedding_dim`, `datasets`, and `tasks`
- [ ] `model.yaml` includes `paper`, `model_card` and `code` links where they exist
- [ ] `preprocessing` is one of the supported options (`turboprep` by default; `cat12` if needed)
- [ ] `datasets` and `tasks` list only supported values (datasets: `NKI`, `HBN`; tasks: `sex`, `age`, `bmi`)

### For code submissions
- [ ] `extract.py` defines `extract(input_dir, output_csv, weights_dir)` with exactly those three arguments
- [ ] `extract.py` reads volumes from `<input_dir>/<subject_id>/normalized.nii.gz` and writes `subject_id,f0,...,fN` to `output_csv`
- [ ] The number of feature columns matches `embedding_dim` in `model.yaml`
- [ ] `weights.txt` lists direct-download URL(s) — HuggingFace (`/resolve/`), Zenodo, GitHub Releases, or an institutional raw link
- [ ] Weight URLs are **not** Google Drive / Dropbox / `/blob/` viewer pages
- [ ] Extraction is frozen-feature only (a forward pass / embedding — no training, seeds set for determinism)

### Dependencies
- [ ] My model runs in the shared evaluation environment (PyTorch + MONAI + nibabel + numpy + ...), **or**
- [ ] I included a `requirements.txt` with any extra dependencies my model needs
- [ ] Any custom package is installable from a public source (e.g. `pip install git+https://...` or PyPI)

### Sanity check
- [ ] I ran `python scripts/validate_extract.py models/<my-model>` locally and it passed
- [ ] Weights are publicly downloadable (no login wall)

---

## Notes for the maintainer

<!-- Anything to flag: unusual dependencies, expected runtime, special handling, etc. -->

> **Note:** Extraction runs on the maintainer's allocation, so a maintainer will review and approve this PR before any cluster job is launched.
