#!/usr/bin/env python3
"""
Score one model folder against everything its model.yaml declares.

Convention:
  models/<name>/model.yaml         # name, datasets, tasks
  models/<name>/features/<DS>.csv  # features for dataset DS (csv or parquet)
  labels/<DS>.csv                  # shared labels for dataset DS

Usage:
  python scripts/score_all.py models/uncnn
  python scripts/score_all.py models/uncnn HBN age   # one pair -> results.HBN.age.json
"""
import json
import os
import sys
import yaml

# make eval/scorer.py importable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval"))
from scorer import score, TASK_TYPE  # noqa: E402


def find_features(model_dir, dataset):
    for ext in (".csv", ".parquet"):
        p = os.path.join(model_dir, "features", dataset + ext)
        if os.path.isfile(p):
            return p
    return None


def main():
    if len(sys.argv) not in (2, 4):
        print("usage: score_all.py <model_dir> [<dataset> <task>]")
        return 2
    model_dir = sys.argv[1].rstrip("/")
    only = sys.argv[2:4]

    meta = yaml.safe_load(open(os.path.join(model_dir, "model.yaml")))
    name = meta.get("name", os.path.basename(model_dir))
    datasets = meta.get("datasets", [])
    tasks = meta.get("tasks", [])
    if not datasets or not tasks:
        print(f"{model_dir}/model.yaml must list 'datasets' and 'tasks'")
        return 1
    bad = [t for t in tasks if t not in TASK_TYPE]
    if bad:
        print(f"unknown task(s) {bad}; allowed: {list(TASK_TYPE)}")
        return 1
    if only:
        if only[0] not in datasets or only[1] not in tasks:
            print(f"{only[0]}/{only[1]} is not declared in {model_dir}/model.yaml")
            return 1
        datasets, tasks = [only[0]], [only[1]]

    rows, failures = [], []
    for ds in datasets:
        feats = find_features(model_dir, ds)
        labels = os.path.join(REPO, "labels", ds + ".csv")
        if feats is None:
            failures.append(f"{ds}: no features file under {model_dir}/features/")
            continue
        if not os.path.isfile(labels):
            failures.append(f"{ds}: no labels file at labels/{ds}.csv")
            continue
        for task in tasks:
            try:
                rows.append(score(feats, labels, ds, task, model_name=name))
            except Exception as e:
                failures.append(f"{ds}/{task}: {e}")

    print("\n==== RESULTS ====")
    for r in rows:
        print(json.dumps(r))
    out = os.path.join(model_dir, "results.%s.%s.json" % tuple(only) if only else "results.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nWrote {out}  ({len(rows)} rows)")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
