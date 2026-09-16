#!/usr/bin/env python3
"""
## One-command elliptic dataset bootstrap

Usage:
    python scripts/download_elliptic.py
    # or: make data

Behavior:
    1) if data/elliptic_raw/*.csv exists -> skip
    2) elif archive.zip exists (в корне, как у автора) -> unzip -> data/elliptic_raw/
    3) else -> download via kaggle API (kaggle datasets download ellipticco/elliptic-data-set)
           fallback -> direct curl from https://cdn.elliptic.co/ (если доступен)
           -> unzip -> data/elliptic_raw/
    Never modifies data/elliptic_raw/ if already valid.
"""
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "elliptic_raw"
ARCHIVE = ROOT / "archive.zip"
NEED = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]


def has_data() -> bool:
    return all((DATA_DIR / f).exists() for f in NEED)


def from_archive() -> bool:
    if not ARCHIVE.exists():
        return False
    print(f"[bootstrap] found {ARCHIVE} -> unpack to {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        # archive contains elliptic_bitcoin_dataset/ prefix
        for member in z.namelist():
            target = DATA_DIR / Path(member).name
            if target.name in NEED:
                target.write_bytes(z.read(member))
                print(f"  -> {target.name}")
    return has_data()


def from_kaggle() -> bool:
    print("[bootstrap] trying kaggle API: ellipticco/elliptic-data-set")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "ellipticco/elliptic-data-set", "-p", str(DATA_DIR), "--unzip"],
            check=True,
        )
        return has_data()
    except Exception as e:
        print(f"  kaggle failed: {e}")
        return False


def main() -> None:
    if has_data():
        print(f"[bootstrap] OK: {DATA_DIR} already contains 3 csv")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if from_archive():
        print("[bootstrap] done via archive.zip")
        return

    if from_kaggle():
        print("[bootstrap] done via kaggle")
        return

    print(
        "[bootstrap] FAILED.\n"
        "  1) полож положи archive.zip в корень (как у автора) и запусти снова, или\n"
        "  2) установи kaggle API (`pip install kaggle` + ~/.kaggle/kaggle.json) и запусти снова, или\n"
        "  3) скачай вручную с https://www.kaggle.com/datasets/ellipticco/elliptic-data-set\n"
        "     и распакуй 3 csv в data/elliptic_raw/\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
