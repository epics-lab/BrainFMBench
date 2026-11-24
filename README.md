[![Paper](https://img.shields.io/badge/Paper-MICCAI_2026_MedAGI-0b3d91?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://papers.miccai.org/miccai-2026-sat/paper/MedAGI_025.pdf)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-2e8b57?style=for-the-badge)](https://www.epics-lab.ca/BrainFMBench/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Space-ffcc4d?style=for-the-badge)](https://huggingface.co/spaces/epics-lab/BrainFMBench)
[![Submissions](https://img.shields.io/badge/Submissions-Open-6a5acd?style=for-the-badge&logo=github&logoColor=white)](CONTRIBUTING.md)
[![Scoring](https://img.shields.io/github/actions/workflow/status/epics-lab/BrainFMBench/score.yml?style=for-the-badge&label=Scoring&logo=githubactions&logoColor=white)](https://github.com/epics-lab/BrainFMBench/actions/workflows/score.yml)

# BrainFMBench

A living benchmark for **structural brain MRI foundation models**. Every model runs as a frozen
encoder on the same subjects from five [Reproducible Brain Charts](https://reprobrainchart.github.io) cohorts,
and the same linear and random forest probes predict sex, age, BMI and mental health scores from its features.
To add a model, open a pull request: see [CONTRIBUTING.md](CONTRIBUTING.md). The leaderboard updates automatically.

Paper: [MICCAI 2026 MedAGI workshop](https://papers.miccai.org/miccai-2026-sat/paper/MedAGI_025.pdf) ·
[bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.05.15.725427v4) ·
[Papers with Code](https://paperswithcode.co/paper/132944) · Code MIT, labels CC BY-NC-SA 4.0 ([details](labels/README.md))

<!-- LEADERBOARD:START -->

[![BrainFMBench overview](leaderboard_overview.png?v=39281c11)](https://www.epics-lab.ca/BrainFMBench/)

**[Interactive version](https://www.epics-lab.ca/BrainFMBench/)**

Full tables with every dataset, task and probe, plus model details: [`LEADERBOARD.md`](LEADERBOARD.md).

<!-- LEADERBOARD:END -->


## Leaderboard

See [`LEADERBOARD.md`](LEADERBOARD.md) for the full table and the [interactive site](https://www.epics-lab.ca/BrainFMBench/)
to switch between the linear and random forest probes and to check sensitivity to rigid versus affine registration.
Scores come from probes on frozen features. The overview heatmap shows the held out test mean over
5 seeds, with the best model in each row boxed; the boxplots show 5 fold CV across 5 seeds as the box and
the held out test results as points. Earlier versions stay browsable on the site; see [`CHANGELOG.md`](CHANGELOG.md).

**Releases are quarterly.** New datasets, tasks and preprocessing options are batched into one release every three months; the next is planned for **March 2027**. New models do not wait for a release: they appear on the leaderboard as soon as their PR is merged. Suggestions for datasets, tasks or models are welcome as an [issue](https://github.com/epics-lab/BrainFMBench/issues/new?template=suggestion.yml) or a PR.

## Tasks & data

Models are evaluated on five datasets from the [Reproducible Brain Charts](https://reprobrainchart.github.io) (RBC)
initiative:

- **NKI** — Nathan Kline Institute Rockland Sample (958 subjects)
- **HBN** — Healthy Brain Network (997 subjects)
- **PNC** — Philadelphia Neurodevelopmental Cohort (1439 subjects)
- **BHRC** — Brazilian High Risk Cohort (542 subjects)
- **CCNP** — Chinese Color Nest Project (178 subjects)

Labels are derived from RBC and distributed under CC BY-NC-SA 4.0; see [labels/README.md](labels/README.md).

Tasks: **sex** classification (balanced accuracy and AUC), **age** / **BMI** regression (mean absolute
error and R²), the RBC harmonized **p factor**, **internalizing** and **externalizing** scores as regression
(MAE and R²) on all five datasets, p factor zero vs nonzero classification (balanced accuracy and AUC) on PNC, and
**handedness**, left vs right, as classification (balanced accuracy and AUC) on PNC, HBN and NKI. Inputs are offered with four preprocessing options:
[turboprep](https://github.com/LemuelPuglisi/turboprep) with rigid registration (default), turboprep with
affine registration to MNI 1 mm, [CAT12](https://neuro-jena.github.io/cat/) gray matter maps, and the raw T1w
for models with their own pipeline.

## How it works

BrainFMBench splits the work between a computing cluster and GitHub actions:

```
  contributor PR                     Alliance cluster (Rorqual)        GitHub Actions
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
labels/<DS>.csv         shared task labels per dataset (see labels/README.md)
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
- **`weights.txt`** — direct-download URL(s) to your checkpoint(s) (Hugging Face,
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
- Two probes on the same splits: a RandomForest (200 trees, depth 6) and a linear probe
  (ridge for regression, L2 logistic regression for classification, regularization chosen
  by inner 5 fold CV on the training data)
- Classification reports balanced accuracy and AUC; regression reports MAE and R²

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
