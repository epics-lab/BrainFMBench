# Changelog

BrainFMBench follows the release policy in [CONTRIBUTING.md](CONTRIBUTING.md#versioning).

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
