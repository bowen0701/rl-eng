#!/usr/bin/env python3
"""
scripts/promote_run_to_stable.py
===============================
Graduates an experimental training run to the 'models/stable' Model Zoo.

Scalable Infra Features:
1. Dynamic Discovery: Reads config.yaml to determine the environment/model name.
2. Artifact Agnostic: Copies all files from the run directory (scalable for DQN, PPO, etc.).
3. Versioning: Supports industry-standard v0.x (experimental) and v1.x (production).

Usage:
    python3 scripts/promote_run_to_stable.py --run_id <id> [--major] [--version <x.y>]
"""

import argparse
import json
import os
import re
import shutil
from datetime import datetime

STABLE_DIR = "models/stable"
RUNS_DIR = "runs"

def get_config_env(run_path: str) -> str:
    """Extracts the 'env' name from config.yaml."""
    config_path = os.path.join(run_path, "config.yaml")
    if not os.path.exists(config_path):
        return ""
    
    try:
        # Simple line-based parse to avoid heavy yaml dependency in graduation script
        with open(config_path, 'r') as f:
            for line in f:
                if line.strip().startswith("env:"):
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return ""

def get_next_version(model_name: str, bump_major: bool = False) -> str:
    """Calculates next version based on existing folders in stable/."""
    pattern = re.compile(rf"^{model_name}_v(\d+)\.(\d+)$")
    h_maj, h_min = 0, 0
    found = False

    if os.path.exists(STABLE_DIR):
        for item in os.listdir(STABLE_DIR):
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
    parser = argparse.ArgumentParser(description="Promote RL run to stable.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--major", action="store_true", help="Bump to next major version.")
    parser.add_argument("--version", help="Manual version override.")
    args = parser.parse_args()

    run_path = os.path.join(RUNS_DIR, args.run_id)
    if not os.path.isdir(run_path):
        print(f"[error] Run directory not found: {run_path}"); return

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
    target_path = os.path.join(STABLE_DIR, target_dir_name)

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
    
    if artifact_count == 0:
        print("[warn] No files were found to graduate!"); shutil.rmtree(target_path); return

    # 4. Persistence Metadata
    meta = {
        "original_run_id": args.run_id,
        "graduation_timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "version": f"v{ver}",
        "files_graduated": artifact_count
    }
    with open(os.path.join(target_path, "graduation_metadata.json"), 'w') as f:
        json.dump(meta, f, indent=4)

    print(f"\n[success] {model_name} graduated to v{ver} ({artifact_count} files)")

if __name__ == "__main__":
    main()
