# Tasks

Every task is predicted from one frozen feature vector per subject, with the same probes for every model. A task runs on a dataset only when that dataset's label file has the column.

| Task | Type | Reported | Datasets |
|---|---|---|---|
| **Sex** | classification | balanced accuracy, AUC | all five |
| **Age** | regression | MAE in years, R² | all five |
| **BMI** | regression | MAE, R² | all five |
| **p factor** | regression | MAE, R² | all five |
| **p factor ≠ 0** | classification | balanced accuracy, AUC | PNC |
| **Internalizing** | regression | MAE, R² | all five |
| **Externalizing** | regression | MAE, R² | all five |
| **Handedness** | classification, left vs right | balanced accuracy, AUC | PNC, HBN, NKI |

## Label definitions

`sex`
:   1 for female, 0 for male.

`p_factor`, `internalizing`, `externalizing`
:   The Reproducible Brain Charts harmonized scores, `*_mcelroy_harmonized_all_samples`. In NKI they exist for 139 subjects only, because the CBCL they come from covers children.

`p_factor_nonzero`
:   1 when the harmonized p factor is above its floor, meaning at least one endorsed symptom, and 0 at the floor.

`handedness`
:   1 for left, 0 for right. Included only where at least 50 subjects are left handed.

The full source and license of every label file is in [`labels/README.md`](https://github.com/epics-lab/BrainFMBench/blob/main/labels/README.md).

## Chance level

Every heatmap cell is compared with chance: 0.5 for balanced accuracy and AUC, 0 for R², and the MAE of a model that always predicts the training mean for MAE. A row counts toward the ranking only when some model clears chance by a clear margin, see [Protocol](protocol.md#ranking).
