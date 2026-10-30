[![MICCAI 2026](https://img.shields.io/badge/MICCAI_2026-MedAGI_Workshop-0b3d91?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://papers.miccai.org/miccai-2026-sat/paper/MedAGI_025.pdf)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-Preprint-b31b1b?style=for-the-badge)](https://www.biorxiv.org/content/10.64898/2026.05.15.725427v4)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-2e8b57?style=for-the-badge)](https://www.epics-lab.ca/BrainFMBench/)
[![Submissions](https://img.shields.io/badge/Submissions-Open-6a5acd?style=for-the-badge&logo=github&logoColor=white)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-MIT-0a7bbb?style=for-the-badge)](LICENSE)
[![Scoring](https://img.shields.io/github/actions/workflow/status/epics-lab/BrainFMBench/score.yml?style=for-the-badge&label=Scoring&logo=githubactions&logoColor=white)](https://github.com/epics-lab/BrainFMBench/actions/workflows/score.yml)
[![Models](https://img.shields.io/badge/%F0%9F%A7%A0_Models-9-e67e22?style=for-the-badge)](LEADERBOARD.md)

# BrainFMBench

A living benchmark for **brain structural MRI foundation models**. Contributors
submit a model; model is applied to MRI datasets to extract features; features are
fed into downstream learners; the leaderboard updates automatically.

<!-- LEADERBOARD:START -->

[![Leaderboard boxplots](leaderboard_boxplots.png)](https://www.epics-lab.ca/BrainFMBench/)

**[Interactive version](https://www.epics-lab.ca/BrainFMBench/)**

| Model | Dataset | Sex (acc) | Age (MAE) | BMI (MAE) |
|-------|---------|-----------|-----------|-----------|
| un-CNN | NKI | 0.857 ± 0.032 | 5.11 ± 0.22 | 3.13 ± 0.11 |
| SwinBrain | NKI | 0.855 ± 0.023 | 9.47 ± 0.54 | 3.97 ± 0.20 |
| 3D-Neuro-SimCLR | NKI | 0.835 ± 0.023 | 5.86 ± 0.58 | 3.52 ± 0.17 |
| BrainIAC | NKI | 0.810 ± 0.031 | 13.96 ± 0.59 | 4.16 ± 0.25 |
| un-CNN | HBN | 0.772 ± 0.043 | 1.42 ± 0.08 | 2.70 ± 0.33 |
| FS aparc | NKI | 0.770 ± 0.023 | 9.09 ± 0.63 | 3.99 ± 0.36 |
| 3D-Neuro-SimCLR | HBN | 0.768 ± 0.049 | 1.64 ± 0.12 | 3.02 ± 0.29 |
| AnatCL (Local) | NKI | 0.765 ± 0.024 | 6.19 ± 0.47 | 3.92 ± 0.25 |
| Dinov3D-neuro | NKI | 0.757 ± 0.045 | 8.29 ± 0.57 | 3.69 ± 0.12 |
| SwinBrain | HBN | 0.744 ± 0.038 | 2.39 ± 0.11 | 3.41 ± 0.36 |
| BrainIAC | HBN | 0.744 ± 0.034 | 2.61 ± 0.07 | 3.59 ± 0.49 |
| FS Schaefer | NKI | 0.740 ± 0.016 | 8.29 ± 0.38 | 3.99 ± 0.35 |
| FS aparc | HBN | 0.740 ± 0.034 | 2.13 ± 0.17 | 3.51 ± 0.31 |
| AnatCL (Global) | NKI | 0.738 ± 0.031 | 6.26 ± 0.43 | 3.92 ± 0.22 |
| FS Schaefer | HBN | 0.736 ± 0.028 | 2.16 ± 0.19 | 3.44 ± 0.31 |
| AnatCL (Global) | HBN | 0.723 ± 0.060 | 1.78 ± 0.18 | 3.39 ± 0.33 |
| AnatCL (Local) | HBN | 0.707 ± 0.063 | 1.85 ± 0.10 | 3.34 ± 0.40 |
| Dinov3D-neuro | HBN | 0.696 ± 0.041 | 1.96 ± 0.11 | 3.23 ± 0.27 |

_Ranking varies by task and dataset; there is no single overall winner._

<!-- LEADERBOARD:END -->


## Leaderboard

See [`LEADERBOARD.md`](LEADERBOARD.md) for the current ranking and the figure. Scores are downstream RandomForest probes of frozen
features (5-fold CV across 5 seeds for the box, held-out test for the points).

## Tasks & data

Models are evaluated on two datasets from the [Reproducible Brain Charts](https://reprobrainchart.github.io) (RBC)
initiative:

- **NKI** — Nathan Kline Institute Rockland Sample (958 subjects)
- **HBN** — Healthy Brain Network (997 subjects)

Models are evaluated across three tasks: **sex** classification (balanced accuracy) and **age** / **BMI**
regression (mean absolute error). Preprocessing is [turboprep](https://github.com/LemuelPuglisi/turboprep) by default, with
[CAT12](https://neuro-jena.github.io/cat/) also available.

## How it works

BrainFMBench splits the work between a computing cluster and GitHub actions:

```
  contributor PR                     Compute Canada (rorqual)          GitHub Actions
  ─────────────                      ────────────────────────         ─────────
  model.yaml                                                          validate PR
  extract.py        ── merge ──▶     extract features    ──▶       score features
  weights.txt                        (on preprocessed data)               │
                                                                       leaderboard
                                                                       updates
```

- **Feature extraction runs on the cluster**, against preprocessed data. Only the resulting feature vectors come back.
- **Scoring runs in CI**, publicly and reproducibly, so anyone can verify how the
  leaderboard numbers were produced.

## Repository layout

```
models/<name>/          submitted models (model.yaml + features, or + extract.py)
labels/<DS>.csv         shared task labels per dataset (subject_id, sex, age, bmi)
eval/scorer.py          the downstream probing protocol
scripts/                scoring, leaderboard, and submission validation
cluster/                the async cluster runner (reap/sow) + sbatch template
example-submission/     a minimal, working submission you can copy
.github/workflows/      validation, scoring, and cluster-extraction workflows
```

## Contributing a model

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide. In short, open a PR
adding `models/<your-model>/` with:

- **`model.yaml`** — metadata (name, preprocessing, embedding_dim, datasets, tasks)
- **`extract.py`** — defines `extract(input_dir, output_csv, weights_dir)`
- **`weights.txt`** — direct-download URL(s) to your checkpoint(s) (HuggingFace,
  Zenodo, etc.)
- **`requirements.txt`** — optional; pinned dependencies. If present, extraction
  runs in an isolated virtualenv built from it rather than the shared cluster
  environment.

Copy [`example-submission/`](example-submission/) as a starting point. A PR
validation check runs your `extract.py` on a synthetic volume before anything
touches the cluster. Because extraction runs on our allocation, only
maintainer-reviewed submissions are executed.

## Scoring protocol

For each model / dataset / task, at the full training size:

- **Box** = 5-fold cross-validation across 5 seeds (25 values)
- **Points** = held-out test result for the same 5 seeds
- RandomForest probe (200 trees, depth 6) on standardized features; balanced
  accuracy for sex, MAE for age / BMI.

Dependencies are pinned (scikit-learn 1.7.1, numpy 2.4.2, pandas 2.2.3) so the
leaderboard is reproducible run to run.

## Citation

If you use BrainFMBench, please cite the preprint for now:

```bibtex
@article{encin2026comparative,
  title     = {A Comparative Evaluation of Structural MRI Foundation Models for Age, Sex, and Body-Mass Index Predictions},
  author    = {Encin, Arel and Gilmore, Asa and Rokem, Ariel and Dickie, Erin and Glatard, Tristan},
  journal   = {bioRxiv},
  pages     = {2026.05.15.725427},
  year      = {2026},
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.64898/2026.05.15.725427}
}
```

The paper was presented at the MedAGI workshop at MICCAI 2026
([paper page](https://papers.miccai.org/miccai-2026-sat/MedAGI_025.html)).
Once the proceedings are published, please cite the MICCAI version instead.
