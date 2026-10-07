#!/usr/bin/env python3
"""
PR validation gate for BrainFMBench submissions.

Runs on pull requests (cheap, on the GitHub runner, before any cluster job).
For each models/<slug>/ that looks like a CODE submission (has extract.py), it:

  1. model.yaml parses + has required fields
  2. weights.txt exists, lists >=1 URL, URLs are direct-download (not blob/drive)
  3. extract.py defines extract(input_dir, output_csv, weights_dir)
  4. extract.py RUNS on a tiny synthetic volume and produces a valid CSV
     (subject_id + numeric feature columns, embedding_dim matches model.yaml)

Fails the PR with a clear message if anything is wrong
"""
import sys
import os
import csv
import glob
import inspect
import tempfile
import importlib.util

import numpy as np

REQUIRED_YAML = {"name", "authors", "preprocessing", "embedding_dim", "datasets", "tasks"}
BAD_URL_MARKERS = ("drive.google.com", "dropbox.com", "/blob/", "docs.google.com")


def fail(msg):
    print(f"  FAIL: {msg}")
    return False


def check_yaml(model_dir):
    import yaml
    p = os.path.join(model_dir, "model.yaml")
    if not os.path.isfile(p):
        return fail("no model.yaml")
    meta = yaml.safe_load(open(p))
    missing = REQUIRED_YAML - set(meta or {})
    if missing:
        return fail(f"model.yaml missing fields: {sorted(missing)}")
    return meta


def check_weights(model_dir, meta=None):
    p = os.path.join(model_dir, "weights.txt")
    if not os.path.isfile(p):
        return fail("no weights.txt")
    urls = [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]
    if not urls:
        # allowed only for the example (no real weights); real submissions need >=1
        if os.path.basename(model_dir) == "example-submission" or (meta or {}).get("weightless"):
            return True
        return fail("weights.txt lists no URLs")
    for u in urls:
        if any(m in u for m in BAD_URL_MARKERS):
            return fail(f"weights URL not a direct download: {u}")
        if not u.startswith("http"):
            return fail(f"weights URL malformed: {u}")
    return True


def check_extract_runs(model_dir, meta):
    p = os.path.join(model_dir, "extract.py")
    spec = importlib.util.spec_from_file_location("sub_extract", p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return fail(f"extract.py failed to import: {e}")
    if not hasattr(mod, "extract"):
        return fail("extract.py must define extract(input_dir, output_csv, weights_dir)")
    sig = inspect.signature(mod.extract)
    if len(sig.parameters) != 3:
        return fail(f"extract() must take 3 args (input_dir, output_csv, weights_dir); "
                    f"has {len(sig.parameters)}")

    import nibabel as nib
    tmp = tempfile.mkdtemp()
    idir = os.path.join(tmp, "in"); os.makedirs(idir)
    for sid in ["SUB01", "SUB02"]:
        sd = os.path.join(idir, sid); os.makedirs(sd)
        vol = (np.random.rand(16, 16, 16) * 100).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, np.eye(4)), os.path.join(sd, "normalized.nii.gz"))
    out = os.path.join(tmp, "out.csv")
    wdir = os.path.join(tmp, "weights"); os.makedirs(wdir)
    try:
        mod.extract(idir, out, wdir)
    except Exception as e:
        return fail(f"extract() raised on synthetic input: {e}")
    if not os.path.isfile(out):
        return fail("extract() produced no output CSV")

    with open(out) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return fail("output CSV has no data rows")
    header = rows[0]
    if header[0] != "subject_id":
        return fail("output CSV first column must be 'subject_id'")
    n_feat = len(header) - 1
    exp = int(meta["embedding_dim"])
    if n_feat != exp:
        return fail(f"output has {n_feat} feature cols but model.yaml embedding_dim={exp}")
    # numeric check
    try:
        [float(x) for x in rows[1][1:]]
    except ValueError:
        return fail("feature values are not numeric")
    print(f"  ok: extract() ran, {len(rows)-1} rows x {n_feat} features (matches embedding_dim)")
    return True


def validate(model_dir):
    print(f"validating {model_dir}")
    if not os.path.isfile(os.path.join(model_dir, "extract.py")):
        print("  (no extract.py -> embeddings-only submission, skipping code checks)")
        return True
    meta = check_yaml(model_dir)
    if not meta:
        return False
    ok = check_weights(model_dir, meta)
    ok = check_extract_runs(model_dir, meta) and ok
    return ok


def main():
    targets = sys.argv[1:]
    if not targets:
        targets = [d for d in glob.glob("models/*") if os.path.isdir(d)]
        # also validate the example
        if os.path.isdir("example-submission"):
            targets.append("example-submission")
    all_ok = True
    for t in sorted(targets):
        all_ok = validate(t) and all_ok
    print("\n" + ("ALL SUBMISSIONS VALID" if all_ok else "VALIDATION FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
