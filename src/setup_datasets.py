import os
import shutil
from pathlib import Path

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def ensure_folder(folder):
    folder.mkdir(parents=True, exist_ok=True)


def copy_folder(src, dst):
    if dst.exists():
        print(f"✓ Already exists: {dst.name}")
        return

    print(f"Copying {src.name} ...")
    shutil.copytree(src, dst)
    print("Done.\n")


def download_cremad():
    print("=" * 60)
    print("CREMA-D")
    print("=" * 60)

    path = Path(kagglehub.dataset_download("ejlok1/cremad"))

    source = path / "AudioWAV"

    destination = DATA_DIR / "CREMA-D" / "AudioWAV"

    ensure_folder(DATA_DIR / "CREMA-D")

    copy_folder(source, destination)


def verify_dataset(name):
    folder = DATA_DIR / name

    if folder.exists():
        print(f"✓ {name:<12} FOUND")
    else:
        print(f"✗ {name:<12} NOT FOUND")


def main():

    ensure_folder(DATA_DIR)

    # --------------------
    # Kaggle Downloads
    # --------------------
    download_cremad()

    # --------------------
    # Verification
    # --------------------
    print("\n")
    print("=" * 60)
    print("DATASET STATUS")
    print("=" * 60)

    verify_dataset("RAVDESS")
    verify_dataset("CREMA-D")
    verify_dataset("TESS")
    verify_dataset("SAVEE")
    verify_dataset("EMO-DB")

    print("\nSetup Finished.")


if __name__ == "__main__":
    main()