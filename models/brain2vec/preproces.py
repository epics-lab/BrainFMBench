import os
import shutil
import subprocess
from pathlib import Path

TEMPLATE = "registration/img_template.nii.gz"


def preprocess(raw_dir: str, out_dir: str) -> None:
    """
    We use NiChart_DLMUSE for skull-stripping (s3 masked folder) and flirt for registration.

    Please install NiChart_DLMUSE from source:
        git clone https://github.com/CBICA/NiChart_DLMUSE
        cd NiChart_DLMUSE
        pip install -e .

    And I believe FLS's flirt already exists in your cluster.

    NiChart_DLMUSE, being an nnUNet, is a bit slow. As we describe in the paper
    600 subjects requires ~45 minutes with an A40. FYI, we have a C++ version for both
    preprocessing and inference (end-to-end ~20sec/scan on CPU!) but not yet tested on
    clusters, thus can't be utilized here. If there are any problems with preprocessing 
    speed, we can work on it more.

    Saves the output Brain2Vec aligned scans at: /{out_dir}/brain2vec_aligned_scans
    """
    out_dir = Path(out_dir)
    subprocess.run(
        [
            "NiChart_DLMUSE",
            "-i",
            raw_dir,
            "-o",
            str(out_dir),
            "-d",
            "cpu",
            "--refaced-data",
        ],
        check=True,
    )

    masked_dir = out_dir / "temp_working_dir" / "s3_masked"
    aligned_dir = out_dir / "brain2vec_aligned_scans"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"}
    for img in sorted(masked_dir.glob("*.nii.gz")):
        out = aligned_dir / f"{img.name[:-7]}_brain2vec_aligned.nii.gz"
        subprocess.run(
            ["flirt", "-in", str(img), "-ref", TEMPLATE, "-out", str(out)],
            check=True,
            env=env,
        )

    shutil.rmtree(out_dir / "temp_working_dir", ignore_errors=True)
