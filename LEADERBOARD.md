# BrainFMBench Leaderboard

Downstream probing of frozen features with a linear and a RandomForest probe, held out test. The static figure shows the RandomForest probe; the interactive site switches probe and preprocessing.
Classification reports balanced accuracy and AUC; regression reports MAE and R². Arrows mark the better direction.

[![Leaderboard boxplots](leaderboard_boxplots.png?v=a434d610)](https://www.epics-lab.ca/BrainFMBench/)

_Boxes show 5-fold cross-validation across 5 seeds (25 values); overlaid points show held-out test results for the same 5 seeds._

**[Interactive version](https://www.epics-lab.ca/BrainFMBench/)**

_Held out test, mean ± std over 5 seeds. There is no single overall score; sort by task on the [interactive site](https://www.epics-lab.ca/BrainFMBench/)._

| Probe | Model | Preprocessing | HBN Sex bal acc ↑ | HBN Sex AUC ↑ | HBN Age MAE ↓ | HBN Age R² ↑ | HBN BMI MAE ↓ | HBN BMI R² ↑ | NKI Sex bal acc ↑ | NKI Sex AUC ↑ | NKI Age MAE ↓ | NKI Age R² ↑ | NKI BMI MAE ↓ | NKI BMI R² ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Linear | 3D-Neuro-SimCLR | turboprep | 0.776 ± 0.034 | 0.856 ± 0.023 | 1.37 ± 0.13 | 0.745 ± 0.056 | 2.84 ± 0.23 | 0.338 ± 0.183 | 0.892 ± 0.021 | 0.964 ± 0.013 | 5.85 ± 0.41 | 0.854 ± 0.026 | 3.38 ± 0.21 | 0.316 ± 0.080 |
| Linear | 3D-Neuro-SimCLR | turboprep-affine | 0.784 ± 0.022 | 0.873 ± 0.015 | **1.16 ± 0.11** | **0.819 ± 0.034** | 2.55 ± 0.20 | 0.485 ± 0.073 | 0.861 ± 0.043 | 0.935 ± 0.027 | 5.48 ± 0.40 | 0.868 ± 0.025 | 2.50 ± 0.27 | 0.660 ± 0.061 |
| Linear | AnatCL (Global) | cat12 | 0.783 ± 0.035 | 0.849 ± 0.045 | 1.61 ± 0.10 | 0.661 ± 0.045 | 3.21 ± 0.33 | 0.206 ± 0.061 | 0.799 ± 0.032 | 0.906 ± 0.010 | 6.25 ± 0.17 | 0.837 ± 0.007 | 3.87 ± 0.20 | 0.208 ± 0.031 |
| Linear | AnatCL (Local) | cat12 | 0.762 ± 0.028 | 0.840 ± 0.037 | 1.63 ± 0.07 | 0.642 ± 0.058 | 3.13 ± 0.34 | 0.227 ± 0.055 | 0.804 ± 0.016 | 0.901 ± 0.012 | 6.22 ± 0.23 | 0.836 ± 0.011 | 3.71 ± 0.24 | 0.276 ± 0.021 |
| Linear | Brain2Vec | raw | 0.793 ± 0.037 | 0.876 ± 0.024 | 1.20 ± 0.07 | 0.796 ± 0.037 | 2.52 ± 0.21 | **0.524 ± 0.073** | 0.881 ± 0.031 | 0.958 ± 0.011 | **4.78 ± 0.22** | **0.906 ± 0.014** | **2.35 ± 0.25** | **0.709 ± 0.040** |
| Linear | BrainIAC | turboprep | 0.782 ± 0.041 | 0.854 ± 0.031 | 2.38 ± 0.15 | 0.329 ± 0.057 | 3.18 ± 0.42 | 0.248 ± 0.088 | 0.842 ± 0.036 | 0.924 ± 0.026 | 10.15 ± 0.24 | 0.567 ± 0.051 | 3.87 ± 0.20 | 0.220 ± 0.093 |
| Linear | BrainIAC | turboprep-affine | 0.678 ± 0.048 | 0.737 ± 0.054 | 1.95 ± 0.13 | 0.484 ± 0.048 | 3.10 ± 0.37 | 0.293 ± 0.075 | 0.750 ± 0.045 | 0.827 ± 0.047 | 8.96 ± 0.18 | 0.668 ± 0.021 | 3.64 ± 0.23 | 0.316 ± 0.045 |
| Linear | Dinov3D-neuro | turboprep | 0.727 ± 0.030 | 0.828 ± 0.031 | 1.42 ± 0.09 | 0.730 ± 0.042 | 2.68 ± 0.26 | 0.442 ± 0.061 | 0.846 ± 0.030 | 0.939 ± 0.019 | 6.03 ± 0.23 | 0.844 ± 0.023 | 3.26 ± 0.10 | 0.432 ± 0.055 |
| Linear | Dinov3D-neuro | turboprep-affine | 0.703 ± 0.041 | 0.784 ± 0.026 | 1.43 ± 0.11 | 0.729 ± 0.052 | 2.96 ± 0.21 | 0.282 ± 0.081 | 0.759 ± 0.040 | 0.859 ± 0.014 | 6.47 ± 0.51 | 0.827 ± 0.036 | 3.07 ± 0.26 | 0.511 ± 0.026 |
| Linear | FS aparc | freesurfer | 0.819 ± 0.028 | 0.888 ± 0.033 | 1.91 ± 0.13 | 0.524 ± 0.061 | 3.34 ± 0.26 | 0.183 ± 0.037 | 0.822 ± 0.045 | 0.908 ± 0.027 | 7.69 ± 0.54 | 0.753 ± 0.023 | 3.89 ± 0.26 | 0.250 ± 0.048 |
| Linear | FS Schaefer | freesurfer | 0.833 ± 0.020 | 0.891 ± 0.027 | 1.74 ± 0.14 | 0.589 ± 0.068 | 3.18 ± 0.26 | 0.235 ± 0.037 | 0.834 ± 0.018 | 0.929 ± 0.018 | 6.68 ± 0.30 | 0.816 ± 0.018 | 3.76 ± 0.23 | 0.286 ± 0.032 |
| Linear | SwinBrain | turboprep | 0.788 ± 0.030 | 0.867 ± 0.020 | 1.93 ± 0.21 | 0.503 ± 0.080 | 2.99 ± 0.45 | 0.358 ± 0.059 | 0.876 ± 0.043 | 0.962 ± 0.017 | 8.98 ± 0.76 | 0.657 ± 0.078 | 3.54 ± 0.14 | 0.330 ± 0.097 |
| Linear | SwinBrain | turboprep-affine | 0.770 ± 0.046 | 0.840 ± 0.034 | 1.73 ± 0.10 | 0.598 ± 0.051 | 3.07 ± 0.40 | 0.286 ± 0.065 | 0.815 ± 0.040 | 0.913 ± 0.012 | 7.68 ± 0.22 | 0.750 ± 0.034 | 3.48 ± 0.20 | 0.342 ± 0.021 |
| Linear | un-CNN † | turboprep | **0.856 ± 0.031** | **0.929 ± 0.025** | 1.46 ± 0.08 | 0.720 ± 0.045 | 2.66 ± 0.33 | 0.463 ± 0.051 | 0.913 ± 0.023 | **0.984 ± 0.006** | 4.94 ± 0.18 | 0.889 ± 0.015 | 2.88 ± 0.19 | 0.508 ± 0.136 |
| Linear | un-CNN † | turboprep-affine | 0.856 ± 0.036 | 0.922 ± 0.016 | 1.22 ± 0.15 | 0.798 ± 0.060 | **2.51 ± 0.23** | 0.510 ± 0.041 | **0.935 ± 0.020** | 0.982 ± 0.009 | 5.31 ± 0.73 | 0.878 ± 0.030 | 2.52 ± 0.15 | 0.644 ± 0.044 |
| RandomForest | 3D-Neuro-SimCLR | turboprep | 0.768 ± 0.049 | 0.816 ± 0.034 | 1.64 ± 0.12 | 0.654 ± 0.046 | 3.02 ± 0.29 | 0.336 ± 0.060 | 0.835 ± 0.023 | 0.928 ± 0.010 | 5.86 ± 0.58 | 0.841 ± 0.029 | 3.52 ± 0.17 | 0.341 ± 0.030 |
| RandomForest | 3D-Neuro-SimCLR | turboprep-affine | 0.671 ± 0.050 | 0.730 ± 0.048 | **1.41 ± 0.10** | 0.727 ± 0.045 | 2.93 ± 0.26 | 0.296 ± 0.100 | 0.746 ± 0.058 | 0.846 ± 0.049 | 5.26 ± 0.30 | 0.866 ± 0.021 | 2.71 ± 0.21 | 0.582 ± 0.027 |
| RandomForest | AnatCL (Global) | cat12 | 0.723 ± 0.060 | 0.802 ± 0.049 | 1.78 ± 0.18 | 0.595 ± 0.045 | 3.39 ± 0.33 | 0.148 ± 0.058 | 0.738 ± 0.031 | 0.850 ± 0.028 | 6.26 ± 0.43 | 0.825 ± 0.024 | 3.92 ± 0.22 | 0.183 ± 0.029 |
| RandomForest | AnatCL (Local) | cat12 | 0.707 ± 0.063 | 0.783 ± 0.058 | 1.85 ± 0.10 | 0.556 ± 0.056 | 3.34 ± 0.40 | 0.178 ± 0.049 | 0.765 ± 0.024 | 0.868 ± 0.024 | 6.19 ± 0.47 | 0.823 ± 0.023 | 3.92 ± 0.25 | 0.197 ± 0.019 |
| RandomForest | Brain2Vec | raw | 0.727 ± 0.048 | 0.811 ± 0.039 | 1.49 ± 0.09 | 0.700 ± 0.040 | 3.01 ± 0.27 | 0.293 ± 0.054 | 0.856 ± 0.044 | 0.933 ± 0.021 | **4.83 ± 0.43** | **0.890 ± 0.029** | 2.98 ± 0.28 | 0.533 ± 0.033 |
| RandomForest | BrainIAC | turboprep | 0.744 ± 0.034 | 0.827 ± 0.026 | 2.61 ± 0.07 | 0.200 ± 0.049 | 3.59 ± 0.49 | 0.069 ± 0.072 | 0.810 ± 0.031 | 0.898 ± 0.021 | 13.96 ± 0.59 | 0.269 ± 0.034 | 4.16 ± 0.25 | 0.108 ± 0.025 |
| RandomForest | BrainIAC | turboprep-affine | 0.604 ± 0.065 | 0.632 ± 0.084 | 2.53 ± 0.06 | 0.227 ± 0.020 | 3.52 ± 0.41 | 0.118 ± 0.066 | 0.566 ± 0.043 | 0.648 ± 0.060 | 13.36 ± 0.69 | 0.346 ± 0.037 | 4.01 ± 0.19 | 0.152 ± 0.060 |
| RandomForest | Dinov3D-neuro | turboprep | 0.696 ± 0.041 | 0.781 ± 0.028 | 1.96 ± 0.11 | 0.510 ± 0.026 | 3.23 ± 0.27 | 0.230 ± 0.044 | 0.757 ± 0.045 | 0.864 ± 0.042 | 8.29 ± 0.57 | 0.696 ± 0.067 | 3.69 ± 0.12 | 0.318 ± 0.049 |
| RandomForest | Dinov3D-neuro | turboprep-affine | 0.634 ± 0.028 | 0.672 ± 0.021 | 2.16 ± 0.16 | 0.419 ± 0.035 | 3.39 ± 0.39 | 0.158 ± 0.046 | 0.674 ± 0.047 | 0.780 ± 0.062 | 8.86 ± 0.30 | 0.692 ± 0.049 | 3.72 ± 0.20 | 0.272 ± 0.007 |
| RandomForest | FS aparc | freesurfer | 0.740 ± 0.034 | 0.824 ± 0.020 | 2.13 ± 0.17 | 0.448 ± 0.034 | 3.51 ± 0.31 | 0.115 ± 0.037 | 0.770 ± 0.023 | 0.862 ± 0.026 | 9.09 ± 0.63 | 0.662 ± 0.024 | 3.99 ± 0.36 | 0.193 ± 0.045 |
| RandomForest | FS Schaefer | freesurfer | 0.736 ± 0.028 | 0.829 ± 0.019 | 2.16 ± 0.19 | 0.425 ± 0.062 | 3.44 ± 0.31 | 0.137 ± 0.049 | 0.740 ± 0.016 | 0.862 ± 0.019 | 8.29 ± 0.38 | 0.722 ± 0.016 | 3.99 ± 0.35 | 0.203 ± 0.045 |
| RandomForest | SwinBrain | turboprep | 0.744 ± 0.038 | 0.835 ± 0.029 | 2.39 ± 0.11 | 0.278 ± 0.073 | 3.41 ± 0.36 | 0.146 ± 0.042 | 0.855 ± 0.023 | 0.922 ± 0.016 | 9.47 ± 0.54 | 0.623 ± 0.026 | 3.97 ± 0.20 | 0.165 ± 0.034 |
| RandomForest | SwinBrain | turboprep-affine | 0.661 ± 0.040 | 0.735 ± 0.043 | 2.13 ± 0.05 | 0.425 ± 0.037 | 3.27 ± 0.38 | 0.194 ± 0.031 | 0.684 ± 0.043 | 0.809 ± 0.027 | 9.02 ± 0.31 | 0.654 ± 0.050 | 3.95 ± 0.21 | 0.184 ± 0.027 |
| RandomForest | un-CNN † | turboprep | **0.772 ± 0.043** | **0.836 ± 0.039** | 1.42 ± 0.08 | **0.741 ± 0.017** | **2.70 ± 0.33** | **0.427 ± 0.063** | **0.857 ± 0.032** | **0.951 ± 0.017** | 5.11 ± 0.22 | 0.868 ± 0.018 | 3.13 ± 0.11 | 0.461 ± 0.079 |
| RandomForest | un-CNN † | turboprep-affine | 0.692 ± 0.027 | 0.779 ± 0.022 | 1.44 ± 0.05 | 0.723 ± 0.024 | 2.77 ± 0.21 | 0.384 ± 0.057 | 0.812 ± 0.029 | 0.922 ± 0.028 | 5.37 ± 0.51 | 0.863 ± 0.032 | **2.60 ± 0.15** | **0.615 ± 0.008** |

_Bold: best per column within each probe._

† **un-CNN:** Maintainers' own model.

### Models

| Model | Preprocessing | Dim | Links |
|---|---|---|---|
| 3D-Neuro-SimCLR | turboprep | 512 | [paper](https://arxiv.org/abs/2509.10620) · [code](https://github.com/emilykaczmarek/3D-Neuro-SimCLR) |
| AnatCL (Global) | cat12 | 512 | [paper](https://www.sciencedirect.com/science/article/pii/S0167865525003848) · [code](https://github.com/EIDOSLAB/AnatCL) |
| AnatCL (Local) | cat12 | 512 | [paper](https://www.sciencedirect.com/science/article/pii/S0167865525003848) · [code](https://github.com/EIDOSLAB/AnatCL) |
| Brain2Vec | Own pipeline: NiChart DLMUSE (DLICV skull stripping) + FSL FLIRT to the Brain2Vec template | 768 | [paper](https://arxiv.org/abs/2610.28411) · [model card](https://huggingface.co/cbica/Brain2Vec) · [code](https://github.com/CBICA/Brain2Vec) |
| BrainIAC | turboprep | 768 | [paper](https://www.nature.com/articles/s41593-026-02202-6) · [code](https://github.com/AIM-KannLab/BrainIAC) |
| Dinov3D-neuro | turboprep | 792 | [model card](https://huggingface.co/huggingbrain/Dinov3d-Neuro) |
| FS Schaefer | FreeSurfer v6.0.1 (RBC): Schaefer 400 cortical thickness and surface area | 800 | [paper](https://papers.miccai.org/miccai-2026-sat/paper/MedAGI_025.pdf) |
| FS aparc | FreeSurfer v6.0.1 (RBC): aparc 68 cortical thickness and surface area | 136 | [paper](https://papers.miccai.org/miccai-2026-sat/paper/MedAGI_025.pdf) |
| SwinBrain | turboprep | 384 | [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12155752/) · [code](https://github.com/MAI-Lab-West-China-Hospital/SwinBrain) |
| un-CNN | turboprep | 9664 | [paper](https://www.biorxiv.org/content/10.64898/2026.06.07.730652v1) · [model card](https://huggingface.co/epics-lab/untrained-cnn) · [code](https://github.com/epics-lab/untrained-cnn) |

_Auto-generated from `models/*/results.json`. Do not edit by hand._