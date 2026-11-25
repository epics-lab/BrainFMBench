# Preprocessing

Models choose their input with `preprocessing:` in `model.yaml`. Every option covers the same subjects, so models are compared on identical people.

<div class="grid cards" markdown>

-   :material-axis-arrow:{ .lg } **turboprep**, the default

    ---

    Skull stripped T1w from [turboprep](https://github.com/LemuelPuglisi/turboprep), rigidly registered, intensity normalized. Head size is preserved.

    `<input_dir>/<subject>/normalized.nii.gz`

-   :material-resize:{ .lg } **turboprep-affine**

    ---

    The same pipeline with affine registration to MNI 1 mm, shape 193 × 229 × 193. Head size is normalized away.

    Z scored with background near −60: mask with `mask.nii.gz`, never `img > 0`.

-   :material-brain:{ .lg } **cat12**

    ---

    [CAT12](https://neuro-jena.github.io/cat/) modulated grey matter maps, for models trained on voxel based morphometry.

    `<input_dir>/<subject>/mwp1.nii`

-   :material-file-image-outline:{ .lg } **raw**

    ---

    The unprocessed T1w for models with their own pipeline. Add a `preprocess.py`; it runs once per dataset and its output is cached.

    `<raw_dir>/<subject>/T1w.nii.gz`

</div>

## Registration sensitivity

Several models were also run on the other turboprep registration. The leaderboard shows every model on its own preprocessing by default; the **Registration sensitivity** checkbox on the [interactive site](https://www.epics-lab.ca/BrainFMBench/) switches those models from rigid to affine, which shows how much a result depends on registration.

FreeSurfer baselines use the Reproducible Brain Charts FreeSurfer outputs, with Desikan Killiany and Schaefer 400 regional features.
