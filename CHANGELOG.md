# Changelog

BrainFMBench follows the release policy in [CONTRIBUTING.md](CONTRIBUTING.md#versioning).

## v2.2.0, unreleased

- Datasets: BHRC (Brazilian High Risk Cohort, 542 subjects) and CCNP (Chinese Color Nest Project, 178 subjects) from Reproducible Brain Charts, subjects passing RBC T1w QC, with raw, turboprep, turboprep-affine and CAT12 inputs. BHRC is longitudinal: the first session passing QC is used, with labels from that session
- Tasks on the new datasets: sex, age, BMI and p factor regression
- FreeSurfer aparc and Schaefer 400 features for BHRC and CCNP from the RBC FreeSurfer outputs
- Tasks: internalizing and externalizing regression (RBC harmonized scores, MAE and R²) on all five datasets, and p factor regression on HBN and NKI, so it now runs on all five
- Task: handedness, left vs right, as classification (balanced accuracy and AUC) on PNC, HBN and NKI, the datasets with at least 50 left handed subjects
- Site and README: an overview heatmap of every model on every dataset and task, colored by improvement over chance relative to the best model in the row, with the best model and those within one seed SD of it boxed. It shows balanced accuracy and MAE by default; the site can switch it to AUC and R²
- Brain2Vec is now listed as NiChart-Brain2Vec at its authors' request, with a link to NiChart in the models table. Models can add an optional `project:` link
- Scoring runs one job per group of tasks instead of one job per model and dataset, and the leaderboard keeps the previous result for any job that fails or times out

## v2.1.0, 2026-11-08

- Dataset: PNC from Reproducible Brain Charts, 1439 subjects passing RBC T1w QC, with raw, turboprep, turboprep-affine and CAT12 inputs
- Task: p factor zero vs nonzero on PNC (RBC harmonized p factor at its floor vs above it), reported with balanced accuracy and AUC. A task runs on a dataset only when `labels/<DS>.csv` has its column
- Labels: `labels/README.md` documents the RBC source, the CC BY-NC-SA 4.0 license of the label files, and the citations
- README: static figure restored (it broke with the third dataset), badges for datasets, preprocessing options, probes and version
- Task: p factor regression on PNC (RBC harmonized p factor), reported with MAE and R², next to the zero vs nonzero task
- Existing numbers unchanged except two linear probe sex AUC values that moved by 0.001 on the rescore (3D-Neuro-SimCLR affine on NKI, BrainIAC affine on HBN)

## v2.0.0, 2026-11-06

- Probe: added a linear probe alongside the RandomForest, on the same splits (5 seeds, 5 fold CV, held out test). Ridge for age and BMI, L2 logistic regression for sex, with alpha or C chosen by inner fivefold CV on the training data only
- Every model rescored on both probes; RandomForest numbers are unchanged
- Leaderboard: every task now shows all of its metrics, balanced accuracy and AUC for sex, MAE and R² for age and BMI. These were already computed and saved; only the display changed
- Site: a probe selector, and under the plot an affine registration checkbox (on by default) that only switches the turboprep models between turboprep-affine and rigid turboprep; AnatCL (CAT12), FreeSurfer and Brain2Vec (raw) do not change. Both drive the plot and the table

## v1.1.0, 2026-11-06

- Preprocessing: added `turboprep-affine`, turboprep with affine registration to MNI 1 mm
- Results: un-CNN, Dinov3D neuro, BrainIAC, SwinBrain and 3D Neuro SimCLR on `turboprep-affine`
- AnatCL, BrainIAC, SwinBrain and 3D Neuro SimCLR now ship their extraction code, the same logic that produced their existing features
- Preprocessing: `cat12` inputs staged on the cluster as `<subject_id>/mwp1.nii`
- No existing numbers changed

## v1.0.0, 2026-11-05

First versioned release.

- Datasets: NKI Rockland, 958 subjects, and HBN, 997 subjects; BMI has 956 and 961
- Tasks: sex classification, age regression, BMI regression
- Probe: RandomForest on standardized frozen features, 5 fold CV over 5 seeds plus a held out test set
- Preprocessing options: turboprep, and raw with an optional `preprocess.py`
- Models: AnatCL global, AnatCL local, Brain2Vec, BrainIAC, Dinov3D neuro, 3D Neuro SimCLR, SwinBrain, un-CNN, and the FreeSurfer aparc and Schaefer baselines
