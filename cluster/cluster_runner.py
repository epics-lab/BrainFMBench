#!/usr/bin/env python3
"""
BrainFMBench cluster runner (the async reap/sow orchestrator).
"""
import os
import sys
import urllib.request

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssh_manager import ClusterSSH

REPO = os.getcwd()
CLUSTER_STAGING = "/home/arelbaha/links/scratch/openmribench_ci_runs"   # per-submission workdirs
CLUSTER_DATA = "/home/arelbaha/links/scratch/OpenMRIdatasets"           # uniform data root
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extract_template.sh")
CLUSTER_USER = "arelbaha"

SUPPORTED_PREP = {"turboprep", "cat12"}


def job_name(slug, dataset):
    return f"ombench_{slug}_{dataset}"


def load_submission(model_dir):
    meta = yaml.safe_load(open(os.path.join(model_dir, "model.yaml")))
    slug = os.path.basename(model_dir)
    return {
        "slug": slug,
        "name": meta.get("name", slug),
        "datasets": meta.get("datasets", []),
        "preprocessing": meta.get("preprocessing", "turboprep"),
        "has_extract": os.path.isfile(os.path.join(model_dir, "extract.py")),
        "has_weights": os.path.isfile(os.path.join(model_dir, "weights.txt")),
        "dir": model_dir,
    }


def features_present_in_repo(model_dir, dataset):
    return os.path.isfile(os.path.join(model_dir, "features", dataset + ".csv"))


def download_weights(weights_txt, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    urls = [l.strip() for l in open(weights_txt) if l.strip() and not l.startswith("#")]
    saved = []
    for url in urls:
        fname = url.split("?")[0].split("/")[-1] or "weight.bin"
        out = os.path.join(dest_dir, fname)
        print(f"    downloading {url} -> {fname}")
        urllib.request.urlretrieve(url, out)
        saved.append(out)
    if not saved:
        print(f"    note: {weights_txt} lists no URLs (weightless submission) -- ok")
    return saved


def render_template(slug, dataset, workdir, modeldir, input_dir):
    t = open(TEMPLATE).read()
    return (t.replace("__JOBNAME__", job_name(slug, dataset))
             .replace("__MODEL_SLUG__", slug)
             .replace("__DATASET__", dataset)
             .replace("__WORKDIR__", workdir)
             .replace("__MODELDIR__", modeldir)
             .replace("__INPUT_DIR__", input_dir))


def ensure_venv(ssh, sub, modeldir):
    """Stage requirements.txt; the sbatch job builds the venv from it.

    The robot account's whitelist permits no shell chaining, so the build
    cannot run here."""
    reqs_local = os.path.join(sub["dir"], "requirements.txt")

    if not os.path.isfile(reqs_local):
        ssh.run(f"rm -rf {modeldir}/venv {modeldir}/requirements.txt", check=False)
        print("    ENV: no requirements.txt -> shared environment")
        return

    ssh.scp_up(reqs_local, f"{modeldir}/requirements.txt")
    print("    ENV: staged requirements.txt (venv built in the job)")


def stage_weights(ssh, sub, modeldir):
    """Weights live at model level, so both datasets share one download."""
    # No shell redirects here: the whitelist passes them as literal arguments.
    rc, out, _ = ssh.run(f"ls -A {modeldir}/weights", check=False)
    if rc == 0 and out.strip():
        print("    WEIGHTS: already staged, skipping download")
        return

    local_wdir = os.path.join("/tmp", f"ombench_{sub['slug']}_weights")
    os.makedirs(local_wdir, exist_ok=True)
    download_weights(os.path.join(sub["dir"], "weights.txt"), local_wdir)
    for wf in os.listdir(local_wdir):
        ssh.scp_up(os.path.join(local_wdir, wf), f"{modeldir}/weights/{wf}")


def sow(ssh, sub, dataset):
    """Stage weights + extract.py to the cluster and sbatch the extraction."""
    slug = sub["slug"]
    prep = sub["preprocessing"]
    if prep not in SUPPORTED_PREP:
        print(f"    SKIP sow: unsupported preprocessing '{prep}'")
        return

    modeldir = f"{CLUSTER_STAGING}/{slug}"
    workdir = f"{modeldir}/{dataset}"
    input_dir = f"{CLUSTER_DATA}/{dataset}/{prep}"

    local_stage = os.path.join("/tmp", f"ombench_{slug}_{dataset}")
    os.makedirs(local_stage, exist_ok=True)
    job_sh = os.path.join(local_stage, "job.sh")
    with open(job_sh, "w") as f:
        f.write(render_template(slug, dataset, workdir, modeldir, input_dir))

    ssh.run(f"mkdir -p {workdir} {modeldir}/weights")
    ensure_venv(ssh, sub, modeldir)
    stage_weights(ssh, sub, modeldir)
    ssh.scp_up(os.path.join(sub["dir"], "extract.py"), f"{modeldir}/extract.py")
    ssh.scp_up(job_sh, f"{workdir}/job.sh")

    rc, out, err = ssh.run(f"sbatch --chdir={workdir} {workdir}/job.sh", check=False)
    if rc == 0:
        print(f"    SOW: submitted -> {out.strip()}")
    else:
        print(f"    SOW FAILED: {err.strip()}")


def reap(ssh, sub, dataset):
    """Pull ONLY the features CSV back into the repo."""
    slug = sub["slug"]
    workdir = f"{CLUSTER_STAGING}/{slug}/{dataset}"
    remote_csv = f"{workdir}/{dataset}.csv"
    dest_dir = os.path.join(sub["dir"], "features")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, dataset + ".csv")
    ssh.scp_down(remote_csv, dest)
    print(f"    REAP: pulled {dataset}.csv -> {dest}")


def main():
    dry = '--dry-run' in sys.argv
    if dry:
        print('*** DRY RUN: no sbatch / scp will happen ***')
    ssh = ClusterSSH()
    queued = ssh.squeue_job_names(CLUSTER_USER)
    print(f"jobs in queue: {sorted(queued) or 'none'}")

    model_dirs = sorted(d for d in
                        (os.path.join(REPO, "models", x) for x in os.listdir(os.path.join(REPO, "models")))
                        if os.path.isdir(d))

    reaped_slugs = set()
    for md in model_dirs:
        sub = load_submission(md)
        # only code-submissions (have extract.py + weights.txt) need the cluster;
        # embeddings-only submissions already ship their features.
        if not (sub["has_extract"] and sub["has_weights"]):
            continue
        print(f"\n== {sub['name']} ({sub['slug']}) ==")
        for dataset in sub["datasets"]:
            if features_present_in_repo(md, dataset):
                print(f"  {dataset}: features already in repo -> nothing to do")
                continue
            jn = job_name(sub["slug"], dataset)
            done_flag = f"{CLUSTER_STAGING}/{sub['slug']}/{dataset}/{dataset}.done"
            if ssh.exists(done_flag):
                if dry:
                    print(f'  {dataset}: [dry] would REAP')
                else:
                    reap(ssh, sub, dataset)
                    reaped_slugs.add(sub["slug"])
            elif jn in queued:
                print(f"  {dataset}: job {jn} still in queue -> skip (reap later)")
            else:
                print(f"  {dataset}: no artifact, not queued -> sow")
                if dry:
                    print(f'  {dataset}: [dry] would SOW (download weights, scp, sbatch)')
                else:
                    sow(ssh, sub, dataset)

    slugs = " ".join(sorted(reaped_slugs))
    print(f"\ndone. reaped={slugs or 'none'}")

    # Which models gained features, so the workflow scores only those.
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"reaped={slugs}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
