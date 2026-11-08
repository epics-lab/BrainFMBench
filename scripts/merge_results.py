#!/usr/bin/env python3
"""
Merge the per (dataset, task) results written by parallel CI scoring jobs
into models/<name>/results.json, in the same row order score_all.py uses.

Each part lives at <parts_dir>/part__<model>__<dataset>__<probe>/results.<dataset>.<task>.<probe>.json

Usage:
  python scripts/merge_results.py parts
"""
import glob
import json
import os
import sys
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
from scorer import PROBES  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print("usage: merge_results.py <parts_dir>")
        return 2

    parts = {}
    # each part holds results.<dataset>.<task>.<probe>.json files for one model
    for p in glob.glob(os.path.join(sys.argv[1], "part__*", "results.*.json")):
        model = os.path.basename(os.path.dirname(p)).split("__")[1]
        _, ds, task, probe, _ = os.path.basename(p).split(".")
        parts[(model, ds, task, probe)] = json.load(open(p))

    for model in sorted({m for m, _, _, _ in parts}):
        meta = yaml.safe_load(open(os.path.join(REPO, "models", model, "model.yaml")))
        rows = []
        for ds in meta["datasets"]:
            # same rules as the plan step in score.yml: no features yet, or no label
            # column for this task on this dataset, means no part is expected
            if not any(os.path.isfile(os.path.join(REPO, "models", model, "features", ds + ext))
                       for ext in (".csv", ".parquet")):
                continue
            with open(os.path.join(REPO, "labels", ds + ".csv")) as f:
                cols = f.readline().strip().split(",")
            for task in meta["tasks"]:
                if task not in cols:
                    continue
                for probe in PROBES:
                    if (model, ds, task, probe) not in parts:
                        print(f"missing part for {model} {ds}/{task}/{probe}")
                        return 1
                    rows.extend(parts[(model, ds, task, probe)])
        out = os.path.join(REPO, "models", model, "results.json")
        json.dump(rows, open(out, "w"), indent=2)
        print(f"Wrote {out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
