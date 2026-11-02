# BrainFMBench Leaderboard

Downstream probing of frozen features (RandomForest, held-out test).
Sex = balanced accuracy (higher is better); Age / BMI = MAE (lower is better).

[![Leaderboard boxplots](leaderboard_boxplots.png?v=37d59cd5)](https://www.epics-lab.ca/BrainFMBench/)

_Boxes show 5-fold cross-validation across 5 seeds (25 values); overlaid points show held-out test results for the same 5 seeds._

**[Interactive version](https://www.epics-lab.ca/BrainFMBench/)**

_Held out test, mean ± std over 5 seeds. Rows sorted by average rank across datasets; best per dataset in bold. There is no single overall score._

#### Sex classification (balanced accuracy, higher is better)

| Avg rank | Model | HBN | NKI |
|---|---|---|---|
| 1.0 | un-CNN † | **0.772 ± 0.043** | **0.857 ± 0.032** |
| 3.0 | 3D-Neuro-SimCLR | 0.768 ± 0.049 | 0.835 ± 0.023 |
| 3.0 | SwinBrain | 0.744 ± 0.038 | 0.855 ± 0.023 |
| 4.5 | Brain2Vec | 0.727 ± 0.048 | 0.856 ± 0.044 |
| 4.5 | BrainIAC | 0.744 ± 0.034 | 0.810 ± 0.031 |
| 5.5 | FS aparc | 0.740 ± 0.034 | 0.770 ± 0.023 |
| 7.5 | FS Schaefer | 0.736 ± 0.028 | 0.740 ± 0.016 |
| 8.0 | AnatCL (Local) | 0.707 ± 0.063 | 0.765 ± 0.024 |
| 9.0 | AnatCL (Global) | 0.723 ± 0.060 | 0.738 ± 0.031 |
| 9.0 | Dinov3D-neuro | 0.696 ± 0.041 | 0.757 ± 0.045 |

#### Age prediction (MAE in years, lower is better)

| Avg rank | Model | HBN | NKI |
|---|---|---|---|
| 1.5 | Brain2Vec | 1.49 ± 0.09 | **4.83 ± 0.43** |
| 1.5 | un-CNN † | **1.42 ± 0.08** | 5.11 ± 0.22 |
| 3.0 | 3D-Neuro-SimCLR | 1.64 ± 0.12 | 5.86 ± 0.58 |
| 4.5 | AnatCL (Global) | 1.78 ± 0.18 | 6.26 ± 0.43 |
| 4.5 | AnatCL (Local) | 1.85 ± 0.10 | 6.19 ± 0.47 |
| 6.5 | Dinov3D-neuro | 1.96 ± 0.11 | 8.29 ± 0.57 |
| 7.0 | FS Schaefer | 2.16 ± 0.19 | 8.29 ± 0.38 |
| 7.5 | FS aparc | 2.13 ± 0.17 | 9.09 ± 0.63 |
| 9.0 | SwinBrain | 2.39 ± 0.11 | 9.47 ± 0.54 |
| 10.0 | BrainIAC | 2.61 ± 0.07 | 13.96 ± 0.59 |

#### BMI prediction (MAE, lower is better)

| Avg rank | Model | HBN | NKI |
|---|---|---|---|
| 1.5 | Brain2Vec | 3.01 ± 0.27 | **2.98 ± 0.28** |
| 1.5 | un-CNN † | **2.70 ± 0.33** | 3.13 ± 0.11 |
| 3.0 | 3D-Neuro-SimCLR | 3.02 ± 0.29 | 3.52 ± 0.17 |
| 4.0 | Dinov3D-neuro | 3.23 ± 0.27 | 3.69 ± 0.12 |
| 5.5 | AnatCL (Global) | 3.39 ± 0.33 | 3.92 ± 0.22 |
| 5.5 | AnatCL (Local) | 3.34 ± 0.40 | 3.92 ± 0.25 |
| 7.0 | SwinBrain | 3.41 ± 0.36 | 3.97 ± 0.20 |
| 8.5 | FS Schaefer | 3.44 ± 0.31 | 3.99 ± 0.35 |
| 8.5 | FS aparc | 3.51 ± 0.31 | 3.99 ± 0.36 |
| 10.0 | BrainIAC | 3.59 ± 0.49 | 4.16 ± 0.25 |

† **un-CNN:** Maintainers' own model. Random features from an untrained 3D CNN: no weights are learned. It extends a classical 3D CNN with multi-channel inputs, a hierarchical encoder with multi-scale feature aggregation, and covariance pooling. The architecture was optimized on NKI, so NKI is not a held out test for un-CNN; it then generalized to HBN and PPMI. PPMI is not on the leaderboard due to data access restrictions. See the paper for the architecture and full results.

### Models

| Model | Preprocessing | Dim | Links |
|---|---|---|---|
| 3D-Neuro-SimCLR | turboprep | 512 | [paper](https://arxiv.org/abs/2509.10620) |
| AnatCL (Global) | cat12 | 512 | [paper](https://www.sciencedirect.com/science/article/pii/S0167865525003848) · [code](https://github.com/EIDOSLAB/AnatCL) |
| AnatCL (Local) | cat12 | 512 | [paper](https://www.sciencedirect.com/science/article/pii/S0167865525003848) · [code](https://github.com/EIDOSLAB/AnatCL) |
| Brain2Vec | raw | 768 | [paper](https://arxiv.org/abs/2610.28411) · [model card](https://huggingface.co/cbica/Brain2Vec) · [code](https://github.com/CBICA/Brain2Vec) |
| BrainIAC | turboprep | 768 | [paper](https://www.nature.com/articles/s41593-026-02202-6) · [code](https://github.com/AIM-KannLab/BrainIAC) |
| Dinov3D-neuro | turboprep | 792 | [model card](https://huggingface.co/huggingbrain/Dinov3d-Neuro) |
| FS Schaefer | freesurfer | 800 | - |
| FS aparc | freesurfer | 136 | - |
| SwinBrain | turboprep | 384 | [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12155752/) |
| un-CNN | turboprep | 9664 | [paper](https://www.biorxiv.org/content/10.64898/2026.06.07.730652v1) · [model card](https://huggingface.co/epics-lab/untrained-cnn) · [code](https://github.com/epics-lab/untrained-cnn) |

_Auto-generated from `models/*/results.json`. Do not edit by hand._