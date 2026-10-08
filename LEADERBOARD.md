# OpenMRIBench Leaderboard

Downstream probing of frozen features (RandomForest, held-out test).
Sex = balanced accuracy (higher is better); Age / BMI = MAE (lower is better).

![Leaderboard boxplots](leaderboard_boxplots.png)

_Boxes show 5-fold cross-validation across 5 seeds (25 values); overlaid points show held-out test results for the same 5 seeds._

| Model | Dataset | Sex (acc) | Age (MAE) | BMI (MAE) |
|-------|---------|-----------|-----------|-----------|
| un-CNN | NKI | 0.857 ± 0.032 | 5.11 ± 0.22 | 3.34 ± 0.21 |
| SwinBrain | NKI | 0.855 ± 0.023 | 9.47 ± 0.54 | 4.18 ± 0.24 |
| 3D-Neuro-SimCLR | NKI | 0.835 ± 0.023 | 5.86 ± 0.58 | 3.83 ± 0.19 |
| BrainIAC | NKI | 0.810 ± 0.031 | 13.96 ± 0.59 | 4.39 ± 0.30 |
| FS aparc | NKI | 0.770 ± 0.023 | 9.09 ± 0.63 | 4.20 ± 0.26 |
| AnatCL (Local) | NKI | 0.765 ± 0.024 | 6.19 ± 0.47 | 4.09 ± 0.15 |
| un-CNN | HBN | 0.760 ± 0.064 | 1.51 ± 0.07 | 2.78 ± 0.26 |
| 3D-Neuro-SimCLR | HBN | 0.749 ± 0.051 | 1.74 ± 0.14 | 3.11 ± 0.34 |
| SwinBrain | HBN | 0.748 ± 0.046 | 2.42 ± 0.21 | 3.49 ± 0.35 |
| FS Schaefer | NKI | 0.740 ± 0.016 | 8.29 ± 0.38 | 4.15 ± 0.33 |
| AnatCL (Global) | NKI | 0.738 ± 0.031 | 6.26 ± 0.43 | 4.09 ± 0.20 |
| BrainIAC | HBN | 0.737 ± 0.036 | 2.77 ± 0.21 | 3.71 ± 0.41 |
| FS Schaefer | HBN | 0.734 ± 0.051 | 2.32 ± 0.17 | 3.51 ± 0.45 |
| FS aparc | HBN | 0.726 ± 0.042 | 2.16 ± 0.18 | 3.60 ± 0.48 |
| AnatCL (Global) | HBN | 0.721 ± 0.062 | 1.91 ± 0.10 | 3.44 ± 0.40 |
| AnatCL (Local) | HBN | 0.703 ± 0.055 | 1.91 ± 0.10 | 3.45 ± 0.43 |

_Auto-generated from `models/*/results.json`. Do not edit by hand._