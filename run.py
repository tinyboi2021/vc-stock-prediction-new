#!/usr/bin/env python3
"""
================================================================================
  VC Stock Prediction — Interactive Run Launcher
================================================================================
  Run this script to choose your execution target:
    1. Local GPU  (your NVIDIA RTX 4060 Ti)
    2. Kaggle P100 (via Kaggle CLI — auto-uploads dataset if needed)

  Usage:
    python run.py
================================================================================
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
import textwrap
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these to match your Kaggle account and project
# ─────────────────────────────────────────────────────────────────────────────

KAGGLE_USERNAME   = "hareeshks"           # Your Kaggle username
KAGGLE_KERNEL_ID  = "stock-prediction-model"     # Slug for the kernel (no spaces)
MAIN_SCRIPT       = "vcStockPredictionEnsemble.py"

# Path to the dataset used by the main script (relative to this file)
# Change this whenever you switch datasets — the launcher detects changes
# automatically and re-uploads only when the file has changed.
DATASET_FILE      = "data/aapl_stock_sentiment_old_dataset_2017_2024.csv"

# Kaggle Dataset slug where your CSV will be uploaded
# Format: "<your-username>/<dataset-slug>"
KAGGLE_DATASET_ID = f"{KAGGLE_USERNAME}/vc-stock-data"
DATASET_SLUG      = "vc-stock-data"

# Local cache file storing the hash of the last uploaded dataset
DATASET_HASH_CACHE = ".last_dataset_hash"

# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.resolve()

CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
BOLD    = "\033[1m"
RESET   = "\033[0m"


def banner():
    print(f"""
{CYAN}{BOLD}
==========================================================
        VC Stock Prediction — Training Launcher            
==========================================================
{RESET}""")


def prompt_choice():
    print(f"  {BOLD}Where do you want to run?{RESET}\n")
    print(f"  {GREEN}[1]{RESET} Local GPU  (NVIDIA RTX 4060 Ti — instant, results saved locally)")
    print(f"  {YELLOW}[2]{RESET} Kaggle P100 (cloud — frees up your PC, 16 GB VRAM)")
    print()
    while True:
        choice = input(f"  Enter choice {BOLD}[1/2]{RESET}: ").strip()
        if choice in ("1", "2"):
            return choice
        print(f"  {RED}Invalid choice. Please enter 1 or 2.{RESET}")


def run_local():
    print(f"\n{GREEN}{BOLD} > Running locally on your GPU...{RESET}\n")
    script = BASE_DIR / MAIN_SCRIPT
    cmd = [sys.executable, str(script)]
    try:
        subprocess.run(cmd, cwd=str(BASE_DIR), check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n{RED}✗ Script exited with error code {e.returncode}.{RESET}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠ Interrupted by user.{RESET}")
        sys.exit(0)


def check_kaggle_cli():
    if shutil.which("kaggle") is None:
        print(f"\n{RED}✗ Kaggle CLI not found.{RESET}")
        print("  Install it with:  pip install kaggle")
        print("  Then place your API token at:  C:\\Users\\<you>\\.kaggle\\kaggle.json")
        sys.exit(1)


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_cached_hash() -> str:
    cache = BASE_DIR / DATASET_HASH_CACHE
    if cache.exists():
        return cache.read_text().strip()
    return ""


def save_hash(md5: str):
    (BASE_DIR / DATASET_HASH_CACHE).write_text(md5)


def dataset_needs_upload(dataset_path: Path) -> bool:
    current_hash = file_md5(dataset_path)
    cached_hash  = get_cached_hash()
    if current_hash != cached_hash:
        print(f"  {YELLOW}Dataset has changed (or is new). Will upload to Kaggle.{RESET}")
        return True
    print(f"  {GREEN}Dataset unchanged — skipping upload.{RESET}")
    return False


def upload_dataset_to_kaggle(dataset_path: Path):
    """
    Creates (or updates) a Kaggle Dataset containing the CSV.
    Steps:
      1. Writes a dataset-metadata.json in a temp staging folder
      2. Copies the CSV into it
      3. Calls `kaggle datasets create` or `kaggle datasets version`
    """
    staging_dir = BASE_DIR / ".kaggle_dataset_staging"
    staging_dir.mkdir(exist_ok=True)

    # Copy the CSV into staging
    dest_csv = staging_dir / dataset_path.name
    shutil.copy2(dataset_path, dest_csv)

    # Write Kaggle dataset-metadata.json
    meta = {
        "title": "VC Stock Prediction Data",
        "id": KAGGLE_DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}]
    }
    meta_file = staging_dir / "dataset-metadata.json"
    meta_file.write_text(json.dumps(meta, indent=2))

    print(f"\n{YELLOW}▶ Uploading dataset '{dataset_path.name}' to Kaggle...{RESET}")

    # Try to create; if it already exists, push a new version instead
    create_cmd = ["kaggle", "datasets", "create", "-p", str(staging_dir), "--dir-mode", "zip"]
    version_cmd = ["kaggle", "datasets", "version", "-p", str(staging_dir),
                   "-m", f"Updated: {dataset_path.name}", "--dir-mode", "zip"]

    result = subprocess.run(create_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  {GREEN}✓ Dataset created successfully.{RESET}")
    elif "already exists" in result.stderr.lower() or "already exists" in result.stdout.lower():
        print(f"  Dataset already exists — pushing a new version...")
        result2 = subprocess.run(version_cmd, capture_output=True, text=True)
        if result2.returncode != 0:
            print(f"  {RED}✗ Version push failed:{RESET}\n{result2.stderr}")
            sys.exit(1)
        print(f"  {GREEN}✓ Dataset version pushed successfully.{RESET}")
    else:
        print(f"  {RED}✗ Upload failed:{RESET}\n{result.stderr}")
        sys.exit(1)

    # Cache the hash so we don't re-upload unnecessarily
    save_hash(file_md5(dataset_path))


def build_kernel_metadata() -> dict:
    """Builds the kernel-metadata.json content."""
    return {
        "id": f"{KAGGLE_USERNAME}/{KAGGLE_KERNEL_ID}",
        "title": "Stock Prediction Model",
        "code_file": MAIN_SCRIPT,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [KAGGLE_DATASET_ID],
        "competition_sources": [],
        "kernel_sources": []
    }


def patch_script_data_path() -> Path:
    """
    Returns the path to a Kaggle-patched copy of the main script where
    DATA_PATH points to the Kaggle dataset mount location
    (/kaggle/input/<dataset-slug>/<filename>).
    The patched copy is written to a temp file alongside the original.
    """
    script_path = BASE_DIR / MAIN_SCRIPT
    patched_path = BASE_DIR / f"_kaggle_{MAIN_SCRIPT}"

    dataset_filename = Path(DATASET_FILE).name
    kaggle_data_path = f"/kaggle/input/{DATASET_SLUG}/{dataset_filename}"

    content = script_path.read_text(encoding="utf-8")

    # Replace the DATA_PATH line with dynamic Kaggle search code
    import re
    dynamic_search_code = f'''
import os
import glob
search_path = glob.glob("/kaggle/input/**/{dataset_filename}", recursive=True)
if search_path:
    DATA_PATH = search_path[0]
else:
    DATA_PATH = "/kaggle/input/{DATASET_SLUG}/{dataset_filename}"
'''.strip()
    
    new_content = re.sub(
        r'^(DATA_PATH\s*=\s*)["\'].*?["\']',
        dynamic_search_code,
        content,
        flags=re.MULTILINE
    )

    if new_content == content:
        print(f"  {YELLOW}WARN Could not find DATA_PATH line to patch — uploading script as-is.{RESET}")
        
    # Kaggle CLI on Windows crashes if the script contains emojis (like 🚨 or 🎯)
    # when reading it with the default 'charmap' (cp1252).
    # Strip all non-ASCII characters before writing the intermediate file.
    safe_content = new_content.encode("ascii", "ignore").decode("ascii")
    
    # Auto-inject pip installations specifically for this Kaggle script!
    pip_injection = (
        "import os\n"
        "os.system('pip install optuna openpyxl mlflow dagshub --quiet')\n\n"
    )
    safe_content = pip_injection + safe_content
    
    patched_path.write_text(safe_content, encoding="utf-8")
    print(f"  {GREEN}OK Script patched: DATA_PATH -> {kaggle_data_path} (emojis stripped, pip injected){RESET}")

    return patched_path

def run_kaggle():
    print(f"\n{YELLOW}{BOLD} > Preparing Kaggle P100 run...{RESET}\n")
    check_kaggle_cli()

    dataset_path = BASE_DIR / DATASET_FILE
    if not dataset_path.exists():
        print(f"  {RED}FAIL Dataset not found at: {dataset_path}{RESET}")
        print(f"  Update DATASET_FILE in run.py to point to your CSV.")
        sys.exit(1)

    # ── Step 1: Upload dataset if it changed ──────────────────────────────
    if dataset_needs_upload(dataset_path):
        upload_dataset_to_kaggle(dataset_path)

    # ── Step 2: Patch the script DATA_PATH for Kaggle ─────────────────────
    patched_script = patch_script_data_path()

    # ── Step 3: Write kernel-metadata.json in a clean staging dir ─────────
    staging_dir = BASE_DIR / ".kaggle_kernel_staging"
    staging_dir.mkdir(exist_ok=True)
    
    # Copy the patched script to the staging dir
    staging_script = staging_dir / patched_script.name
    shutil.copy2(patched_script, staging_script)

    meta = build_kernel_metadata()
    # Point to the patched script name
    meta["code_file"] = patched_script.name
    meta_file = staging_dir / "kernel-metadata.json"
    meta_file.write_text(json.dumps(meta, indent=2))
    print(f"  {GREEN}OK kernel-metadata.json stashed in cleanly.{RESET}")

    # ── Step 4: Push kernel to Kaggle ─────────────────────────────────────
    print(f"\n{YELLOW} > Pushing kernel to Kaggle...{RESET}")
    # Push ONLY the staging directory, not the whole massive project folder!
    push_cmd = ["kaggle", "kernels", "push", "-p", str(staging_dir)]
    result = subprocess.run(push_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  {RED}FAIL Kernel push failed:{RESET}\n{result.stderr}")
        sys.exit(1)
    print(f"  {GREEN}OK Kernel pushed successfully!{RESET}")
    print(f"\n  {CYAN}Monitor:  https://www.kaggle.com/code/{KAGGLE_USERNAME}/{KAGGLE_KERNEL_ID}{RESET}")

    # ── Step 5: Offer to tail the kernel status ────────────────────────────
    print()
    watch = input(f"  Watch live status in terminal? {BOLD}[y/N]{RESET}: ").strip().lower()
    if watch == "y":
        tail_kaggle_status()

    # ── Step 6: Offer to download results when done ───────────────────────
    print()
    dl = input(f"  Download results when finished? {BOLD}[y/N]{RESET}: ").strip().lower()
    if dl == "y":
        download_kaggle_output()


def tail_kaggle_status():
    """Polls the kernel status every 30s until it finishes."""
    import time
    kernel_ref = f"{KAGGLE_USERNAME}/{KAGGLE_KERNEL_ID}"
    print(f"\n  Polling status for '{kernel_ref}' (Ctrl+C to stop)...\n")
    try:
        while True:
            result = subprocess.run(
                ["kaggle", "kernels", "status", kernel_ref],
                capture_output=True, text=True
            )
            output = result.stdout.strip()
            print(f"  [{output}]")
            if any(s in output.lower() for s in ("complete", "error", "cancelled")):
                break
            time.sleep(30)
    except KeyboardInterrupt:
        print(f"\n  {YELLOW}Stopped watching. Kernel still running on Kaggle.{RESET}")


def download_kaggle_output():
    kernel_ref = f"{KAGGLE_USERNAME}/{KAGGLE_KERNEL_ID}"
    # Download directly into the project root so it merges with 
    # ./Saved_Models/, ./.Results/, and ./Prediction_Excel_Sheets/ seamlessly!
    out_dir = BASE_DIR
    print(f"\n{YELLOW} > Downloading Kaggle output to local directories...{RESET}")
    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_ref, "-p", str(out_dir)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  {GREEN}OK Results downloaded successfully!{RESET}")
    else:
        print(f"  {RED}FAIL Download failed:{RESET}\n{result.stderr}")


def main():
    banner()
    choice = prompt_choice()
    if choice == "1":
        run_local()
    else:
        run_kaggle()


if __name__ == "__main__":
    main()
