# BrainFMBench Leaderboard

Downstream probing of frozen features (RandomForest, held-out test).
Sex = balanced accuracy (higher is better); Age / BMI = MAE (lower is better).

[![Leaderboard boxplots](leaderboard_boxplots.png?v=37d59cd5)](https://www.epics-lab.ca/BrainFMBench/)

_Boxes show 5-fold cross-validation across 5 seeds (25 values); overlaid points show held-out test results for the same 5 seeds._

**[Interactive version](https://www.epics-lab.ca/BrainFMBench/)**

_Held out test, mean ± std over 5 seeds. There is no single overall score; sort by task on the [interactive site](https://www.epics-lab.ca/BrainFMBench/)._

| Probe | Model | Preprocessing | HBN Sex (acc ↑) | HBN Age (MAE ↓) | HBN BMI (MAE ↓) | NKI Sex (acc ↑) | NKI Age (MAE ↓) | NKI BMI (MAE ↓) |
|---|---|---|---|---|---|---|---|---|
| RandomForest | 3D-Neuro-SimCLR | turboprep | 0.768 ± 0.049 | 1.64 ± 0.12 | 3.02 ± 0.29 | 0.835 ± 0.023 | 5.86 ± 0.58 | 3.52 ± 0.17 |
| RandomForest | AnatCL (Global) | cat12 | 0.723 ± 0.060 | 1.78 ± 0.18 | 3.39 ± 0.33 | 0.738 ± 0.031 | 6.26 ± 0.43 | 3.92 ± 0.22 |
| RandomForest | AnatCL (Local) | cat12 | 0.707 ± 0.063 | 1.85 ± 0.10 | 3.34 ± 0.40 | 0.765 ± 0.024 | 6.19 ± 0.47 | 3.92 ± 0.25 |
| RandomForest | Brain2Vec | raw | 0.727 ± 0.048 | 1.49 ± 0.09 | 3.01 ± 0.27 | 0.856 ± 0.044 | **4.83 ± 0.43** | **2.98 ± 0.28** |
| RandomForest | BrainIAC | turboprep | 0.744 ± 0.034 | 2.61 ± 0.07 | 3.59 ± 0.49 | 0.810 ± 0.031 | 13.96 ± 0.59 | 4.16 ± 0.25 |
| RandomForest | Dinov3D-neuro | turboprep | 0.696 ± 0.041 | 1.96 ± 0.11 | 3.23 ± 0.27 | 0.757 ± 0.045 | 8.29 ± 0.57 | 3.69 ± 0.12 |
| RandomForest | FS aparc | freesurfer | 0.740 ± 0.034 | 2.13 ± 0.17 | 3.51 ± 0.31 | 0.770 ± 0.023 | 9.09 ± 0.63 | 3.99 ± 0.36 |
| RandomForest | FS Schaefer | freesurfer | 0.736 ± 0.028 | 2.16 ± 0.19 | 3.44 ± 0.31 | 0.740 ± 0.016 | 8.29 ± 0.38 | 3.99 ± 0.35 |
| RandomForest | SwinBrain | turboprep | 0.744 ± 0.038 | 2.39 ± 0.11 | 3.41 ± 0.36 | 0.855 ± 0.023 | 9.47 ± 0.54 | 3.97 ± 0.20 |
| RandomForest | un-CNN † | turboprep | **0.772 ± 0.043** | **1.42 ± 0.08** | **2.70 ± 0.33** | **0.857 ± 0.032** | 5.11 ± 0.22 | 3.13 ± 0.11 |

_Bold: best per column within each probe._

† **un-CNN:** Maintainers' own model.

### Models

| Model | Preprocessing | Dim | Links |
|---|---|---|---|
| 3D-Neuro-SimCLR | turboprep | 512 | [paper](https://arxiv.org/abs/2509.10620) |
| AnatCL (Global) | cat12 | 512 | [paper](https://www.sciencedirect.com/science/article/pii/S0167865525003848) · [code](https://github.com/EIDOSLAB/AnatCL) |
| AnatCL (Local) | cat12 | 512 | [paper](https://www.sciencedirect.com/science/article/pii/S0167865525003848) · [code](https://github.com/EIDOSLAB/AnatCL) |
| Brain2Vec | Own pipeline: NiChart DLMUSE (DLICV skull stripping) + FSL FLIRT to the Brain2Vec template | 768 | [paper](https://arxiv.org/abs/2610.28411) · [model card](https://huggingface.co/cbica/Brain2Vec) · [code](https://github.com/CBICA/Brain2Vec) |
| BrainIAC | turboprep | 768 | [paper](https://www.nature.com/articles/s41593-026-02202-6) · [code](https://github.com/AIM-KannLab/BrainIAC) |
| Dinov3D-neuro | turboprep | 792 | [model card](https://huggingface.co/huggingbrain/Dinov3d-Neuro) |
| FS Schaefer | FreeSurfer v6.0.1 (RBC): Schaefer 400 cortical thickness and surface area | 800 | [paper](https://papers.miccai.org/miccai-2026-sat/paper/MedAGI_025.pdf) |
| FS aparc | FreeSurfer v6.0.1 (RBC): aparc 68 cortical thickness and surface area | 136 | [paper](https://papers.miccai.org/miccai-2026-sat/paper/MedAGI_025.pdf) |
| SwinBrain | turboprep | 384 | [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12155752/) |
| un-CNN | turboprep | 9664 | [paper](https://www.biorxiv.org/content/10.64898/2026.06.07.730652v1) · [model card](https://huggingface.co/epics-lab/untrained-cnn) · [code](https://github.com/epics-lab/untrained-cnn) |

_Auto-generated from `models/*/results.json`. Do not edit by hand._