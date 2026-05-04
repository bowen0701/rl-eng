#!/usr/bin/env python3
"""
scripts/promote_run_to_export.py
===============================
Promotes an experimental training run into `exports/`.

Scalable Infra Features:
1. Dynamic Discovery: Reads config.yaml to determine the environment/model name.
2. Artifact Agnostic: Copies all files from the run directory (scalable for DQN, PPO, etc.).
3. Versioning: Supports industry-standard v0.x (experimental) and v1.x (production).

Usage:
    python3 scripts/promote_run_to_export.py --run_id <id> [--major] [--version <x.y>]
"""

import argparse
import os
import re
import shutil
from datetime import datetime

import yaml # Ensure yaml is imported for dumping

EXPORTS_DIR = "exports" # Changed from "artifacts/exports"
EXPERIMENTS_DIR = "experiments"

def find_run_path(run_id: str) -> str:
    """Finds the full path to a run directory by searching within experiments/."""
    if not os.path.exists(EXPERIMENTS_DIR):
        return ""
    
    for exp_name in os.listdir(EXPERIMENTS_DIR):
        exp_path = os.path.join(EXPERIMENTS_DIR, exp_name)
        if not os.path.isdir(exp_path):
            continue
            
        runs_dir = os.path.join(exp_path, "runs")
        if os.path.exists(runs_dir):
            run_path = os.path.join(runs_dir, run_id)
            if os.path.isdir(run_path):
                return run_path
    return ""

def get_config_env(run_path: str) -> str:
    """Extracts the 'env' name from config.yml (falls back to config.yaml)."""
    for name in ("config.yml", "config.yaml"):
        config_path = os.path.join(run_path, name)
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    if line.strip().startswith("env:"):
                        return line.split(":")[1].strip()
        except Exception:
            pass
    return ""

def get_next_version(model_name: str, bump_major: bool = False) -> str:
    """Calculates next version based on existing folders in exports/."""
    pattern = re.compile(rf"^{model_name}_v(\d+)\.(\d+)$")
    h_maj, h_min = 0, 0
    found = False

    if os.path.exists(EXPORTS_DIR):
        for item in os.listdir(EXPORTS_DIR):
            match = pattern.match(item)
            if match:
                found = True
                maj, min_ = int(match.group(1)), int(match.group(2))
                if maj > h_maj:
                    h_maj, h_min = maj, min_
                elif maj == h_maj and min_ > h_min:
                    h_min = min_

    if not found:
        return "1.0" if bump_major else "0.1"
    
    if bump_major:
        return f"{h_maj + 1}.0"
    return f"{h_maj}.{h_min + 1}"

def main():
    parser = argparse.ArgumentParser(description="Promote an RL run to exports.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--major", action="store_true", help="Bump to next major version.")
    parser.add_argument("--version", help="Manual version override.")
    args = parser.parse_args()

    run_path = find_run_path(args.run_id)
    if not run_path:
        print(f"[error] Run directory not found for run_id: {args.run_id}"); return

    # 1. Discover Identity from Config
    model_name = get_config_env(run_path)
    if not model_name:
        # Fallback to run_id prefix if config is missing or broken
        match = re.match(r"^(.*?)_\d{8}_", args.run_id)
        model_name = match.group(1) if match else args.run_id
    
    print(f"[*] Identified model: {model_name}")

    # 2. Versioning
    ver = args.version.lstrip('v') if args.version else get_next_version(model_name, args.major)
    target_dir_name = f"{model_name}_v{ver}"
    target_path = os.path.join(EXPORTS_DIR, target_dir_name)

    if os.path.exists(target_path):
        print(f"[error] Version v{ver} already exists at {target_path}"); return

    # 3. Graduate all artifacts
    print(f"[*] Promoting all artifacts to {target_path}...")
    os.makedirs(target_path, exist_ok=True)
    
    artifact_count = 0
    for item in os.listdir(run_path):
        s = os.path.join(run_path, item)
        if os.path.isfile(s) and not item.startswith('.'):
            shutil.copy2(s, target_path)
            artifact_count += 1
        elif item == "checkpoints" and os.path.isdir(s):
            shutil.copytree(s, os.path.join(target_path, "checkpoints"))
            artifact_count += len(os.listdir(s))

    if artifact_count == 0:
        print("[warn] No files were found to graduate!"); shutil.rmtree(target_path); return

    # 4. Persistence Metadata
    meta = {
        "original_run_id": args.run_id,
        "promotion_timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "version": f"v{ver}",
        "files_graduated": artifact_count
    }
    with open(os.path.join(target_path, "export_metadata.yaml"), 'w') as f: # Changed to .yaml
        yaml.dump(meta, f, indent=4) # Changed to yaml.dump

    print(f"\n[success] {model_name} promoted to exports v{ver} ({artifact_count} files)")

if __name__ == "__main__":
    main()
