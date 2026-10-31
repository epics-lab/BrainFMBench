#!/bin/bash
#SBATCH --account=rrg-glatard
#SBATCH --cpus-per-task=__CPUS__
#SBATCH --mem=32G
#SBATCH --time=__WALLTIME__
#SBATCH --job-name=__JOBNAME__
#SBATCH --output=__WORKDIR__/extract_%j.out

set -euo pipefail

MODEL_SLUG="__MODEL_SLUG__"
DATASET="__DATASET__"
WORKDIR="__WORKDIR__"          # staging/<slug>/<dataset>: per-dataset outputs
MODELDIR="__MODELDIR__"        # staging/<slug>: venv, weights, extract.py
INPUT_DIR="__INPUT_DIR__"

WEIGHTS_DIR="$MODELDIR/weights"
EXTRACT_PY="$MODELDIR/extract.py"
REQS="$MODELDIR/requirements.txt"
VENV="$MODELDIR/venv"
OUT_CSV="$WORKDIR/${DATASET}.csv"
DONE_FLAG="$WORKDIR/${DATASET}.done"

echo "[$(date)] extraction start: model=$MODEL_SLUG dataset=$DATASET"
echo "  input_dir=$INPUT_DIR"
echo "  weights_dir=$WEIGHTS_DIR"

module load StdEnv/2023 python/3.11

# Both dataset jobs for this model share $MODELDIR, so serialise the build.
exec 200>"$MODELDIR/.venv.lock"
flock 200

if [ -f "$REQS" ]; then
    REQS_HASH="$(sha256sum "$REQS" | cut -d' ' -f1)"
    STAMP="$VENV/.reqs_sha256"

    if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$REQS_HASH" ]; then
        echo "[env] building venv for $MODEL_SLUG"
        rm -rf "$VENV"
        virtualenv --no-download "$VENV"
        source "$VENV/bin/activate"
        pip install --no-index --upgrade pip
        # Compute nodes have no outbound network: wheelhouse only.
        if ! pip install --no-index -r "$REQS"; then
            echo "[env] ERROR: a requirement is not in the Alliance wheelhouse."
            echo "[env] See https://docs.alliancecan.ca/wiki/Available_Python_wheels"
            exit 1
        fi
        echo "$REQS_HASH" > "$STAMP"
    else
        echo "[env] reusing venv (requirements unchanged)"
        source "$VENV/bin/activate"
    fi
else
    echo "[env] no requirements.txt -- using shared module environment"
fi

flock -u 200

echo "[env] python: $(which python)"

# Optional preprocess.py: preprocess(raw_dir, out_dir), cached per dataset.
PREP_PY="$MODELDIR/preprocess.py"
if [ -f "$PREP_PY" ]; then
    PREP_DIR="$WORKDIR/preprocessed"
    PREP_HASH="$(sha256sum "$PREP_PY" | cut -d' ' -f1)"
    if [ "$(cat "$PREP_DIR/preprocess.done" 2>/dev/null)" != "$PREP_HASH" ]; then
        echo "[$(date)] preprocess start: $INPUT_DIR -> $PREP_DIR"
        rm -rf "$PREP_DIR"; mkdir -p "$PREP_DIR"
        python - "$INPUT_DIR" "$PREP_DIR" "$PREP_PY" <<'PYEOT'
import sys, importlib.util
raw_dir, out_dir, prep_py = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("submission_preprocess", prep_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if not hasattr(mod, "preprocess"):
    raise SystemExit("submission preprocess.py must define preprocess(raw_dir, out_dir)")
mod.preprocess(raw_dir, out_dir)
PYEOT
        echo "$PREP_HASH" > "$PREP_DIR/preprocess.done"
        echo "[$(date)] preprocess done"
    else
        echo "[prep] reusing $PREP_DIR (preprocess.py unchanged)"
    fi
    INPUT_DIR="$PREP_DIR"
fi

python - "$INPUT_DIR" "$OUT_CSV" "$WEIGHTS_DIR" "$EXTRACT_PY" <<'PYEOT'
import sys, importlib.util
input_dir, output_csv, weights_dir, extract_py = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("submission_extract", extract_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if not hasattr(mod, "extract"):
    raise SystemExit("submission extract.py must define extract(input_dir, output_csv, weights_dir)")
mod.extract(input_dir, output_csv, weights_dir)
PYEOT

if [ ! -s "$OUT_CSV" ]; then
    echo "ERROR: extract.py produced no output CSV"
    exit 1
fi

# Sentinel; the reaper treats its absence as "not finished".
echo "$(date)" > "$DONE_FLAG"
echo "[$(date)] extraction done: wrote $OUT_CSV and $DONE_FLAG"
