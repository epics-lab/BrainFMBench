"""Brain2Vec preprocessing: NiChart_DLMUSE skull stripping, then FSL flirt to the Brain2Vec template.

NiChart_DLMUSE runs from its own venv (venv_prep next to this file) because it needs torch<=2.2.1.
"""
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "registration" / "img_template.nii.gz"
NICHART = os.environ.get("NICHART_BIN", str(HERE / "venv_prep" / "bin" / "NiChart_DLMUSE"))
FSL = "module load StdEnv/2023 fsl/6.0.7.20 >/dev/null 2>&1"


def preprocess(raw_dir, out_dir):
    out = Path(out_dir)
    flat = out / "flat_input"
    flat.mkdir(parents=True, exist_ok=True)
    for sid in sorted(os.listdir(raw_dir)):
        src = Path(raw_dir, sid, "T1w.nii.gz")
        dst = flat / f"{sid}.nii.gz"
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())

    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    prep_bin = str(Path(NICHART).parent)  # NiChart calls DLICV and DLMUSE by name
    env = {**os.environ, "PATH": prep_bin + os.pathsep + os.environ.get("PATH", "")}
    subprocess.run([NICHART, "-i", str(flat), "-o", str(out / "nichart"), "-d", device, "--refaced-data"],
                   check=True, env=env)

    masked = out / "nichart" / "temp_working_dir" / "s3_masked"
    aligned = out / "brain2vec_aligned_scans"
    aligned.mkdir(exist_ok=True)
    def register(img):
        sid = img.name.split("_")[0].replace(".nii.gz", "")
        subprocess.run(["bash", "-lc", f'{FSL}; FSLOUTPUTTYPE=NIFTI_GZ flirt -in "{img}" -ref "{TEMPLATE}" '
                                       f'-out "{aligned / (sid + ".nii.gz")}"'], check=True)

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(register, sorted(masked.glob("*.nii.gz"))))
    shutil.rmtree(out / "nichart" / "temp_working_dir", ignore_errors=True)
    shutil.rmtree(flat, ignore_errors=True)
