#!/usr/bin/env python3

"""Validate every models/*/model.yaml against the submission schema."""

import sys
from pathlib import Path
import yaml

REQUIRED = {"name": str, "authors": list, "preprocessing": str, "embedding_dim": int}
ALLOWED_PREPROCESSING = {"turboprep", "cat12", "freesurfer", "raw"}

def validate_one(path):
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        return [f"{path}: not valid YAML ({e})"]
    if not isinstance(data, dict):
        return [f"{path}: top level must be a mapping of fields"]
    errors = []
    for field, expected in REQUIRED.items():
        if field not in data:
            errors.append(f"{path}: missing required field '{field}'")
        elif not isinstance(data[field], expected):
            errors.append(f"{path}: '{field}' must be {expected.__name__}, "
                          f"got {type(data[field]).__name__}")
    pp = data.get("preprocessing")
    if isinstance(pp, str) and pp not in ALLOWED_PREPROCESSING:
        errors.append(f"{path}: preprocessing '{pp}' not in {sorted(ALLOWED_PREPROCESSING)}")
    dim = data.get("embedding_dim")
    if isinstance(dim, int) and dim <= 0:
        errors.append(f"{path}: embedding_dim must be positive")
    return errors

def main():
    files = sorted(Path("models").glob("*/model.yaml"))
    if not files:
        print("No models/*/model.yaml found — nothing to validate.")
        return 0
    all_errors = []
    for f in files:
        errs = validate_one(f)
        all_errors += errs
        print(f"{'FAIL' if errs else 'OK'}: {f}")
    if all_errors:
        print("\nValidation failed:")
        for e in all_errors:
            print(f"  - {e}")
        return 1
    print(f"\nAll {len(files)} model.yaml file(s) valid.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
