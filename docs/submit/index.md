# Submit a model

Your model runs as a frozen encoder on our cluster, on every subject, and shows up on the leaderboard as soon as your pull request is merged. No new release needed.

!!! tip "Before you start"
    You need three things: public code that loads your model, weights reachable by a direct download link, and a function that turns one scan into one vector.

## In five steps

1.  **Fork and copy the template**

    [Fork epics-lab/BrainFMBench :material-source-fork:](https://github.com/epics-lab/BrainFMBench/fork){ .md-button }

    Then clone your fork, so your pull request comes from it:

    ```bash
    git clone https://github.com/YOUR-USERNAME/BrainFMBench && cd BrainFMBench
    cp -r example-submission models/my-model
    ```

    Folder names use lowercase letters, digits, `_` and `-`.

2.  **Write `extract.py`**

    ```python
    def extract(input_dir, output_csv, weights_dir):
        # for each <input_dir>/<subject_id>/normalized.nii.gz
        #   load, run a forward pass, pool to one vector
        # write subject_id,f0,...,fN to output_csv
    ```

    Forward pass only, no training. Set your seeds.

3.  **Fill in `model.yaml`**

    ```yaml
    name: My Model
    authors: [Lastname, F., et al.]
    preprocessing: turboprep      # turboprep-affine, cat12 or raw
    embedding_dim: 768
    paper: https://...
    code: https://...
    model_card: https://...
    ```

4.  **List your weights in `weights.txt`**

    One direct download link per line: Hugging Face `/resolve/` links, Zenodo or GitHub Releases. Gated weights are fine, say so in the pull request.

5.  **Open the pull request**

    Validation runs `extract.py` on a synthetic volume. A maintainer reviews, merges, and extraction starts on the cluster. Results usually appear within a day.

## A real example

[Neuro-JEPA-B](../models/neuro-jepa.md) was submitted by its authors in three files: an `extract.py` that averages the final layer tokens into 768 values, a `model.yaml` choosing `turboprep-affine`, and a `weights.txt` pointing to its gated Hugging Face checkpoint. Browse it in [`models/neuro-jepa`](https://github.com/epics-lab/BrainFMBench/tree/main/models/neuro-jepa).

Every detail of the interface is in the [Reference](reference.md).
