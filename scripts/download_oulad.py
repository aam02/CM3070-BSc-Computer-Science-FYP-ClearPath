#!/usr/bin/env python3
"""Download OULAD CSVs into data/raw/."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests

SOURCES = (
    (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00349/OULAD.zip",
        "UCI ML Repository",
    ),
    (
        "https://github.com/Caellwyn/ou_student_predictions/raw/main/content/anonymisedData.zip",
        "GitHub mirror (anonymisedData.zip)",
    ),
)

# assessments / studentAssessment unused.
REQUIRED = (
    "studentVle.csv",
    "vle.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "courses.csv",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    with open(path, "rb") as f:
        return f.read(4) == b"PK\x03\x04"


def _download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=600, allow_redirects=True) as resp:
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").lower()
        if "html" in content_type and "zip" not in content_type:
            raise RuntimeError(f"URL returned HTML, not a zip: {url}")
        total = int(resp.headers.get("content-length") or 0)
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100.0 * downloaded / total
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="")
                else:
                    print(f"\r  {downloaded / 1e6:.1f} MB", end="")
        print()
    if not _is_zip(dest):
        raise RuntimeError(f"Downloaded file is not a zip archive: {dest}")


def main() -> int:
    root = _project_root()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "oulad.zip"

    existing = [p for p in REQUIRED if (raw_dir / p).exists()]
    if len(existing) == len(REQUIRED):
        print("Required OULAD CSVs already present in data/raw/. Skipping download.")
        for name in REQUIRED:
            path = raw_dir / name
            print(f"  {name}: {path.stat().st_size / 1e6:.1f} MB")
        if zip_path.exists():
            zip_path.unlink()
            print("Removed leftover oulad.zip.")
        return 0

    last_error: Exception | None = None
    for url, label in SOURCES:
        try:
            print(f"Downloading OULAD from {label}\n  {url}")
            _download(url, zip_path)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"  failed: {exc}")
            if zip_path.exists():
                zip_path.unlink()
    else:
        print(f"ERROR: all download sources failed. Last error: {last_error}", file=sys.stderr)
        return 1

    print(f"Extracting required CSVs from {zip_path} -> {raw_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        wanted = set(REQUIRED)
        for info in zf.infolist():
            name = Path(info.filename).name
            if name not in wanted:
                continue
            target = raw_dir / name
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"  extracted {name}: {target.stat().st_size / 1e6:.1f} MB")

    missing = [n for n in REQUIRED if not (raw_dir / n).exists()]
    if missing:
        print(f"ERROR: missing files after extract: {missing}", file=sys.stderr)
        return 1

    zip_path.unlink(missing_ok=True)
    print("Removed oulad.zip after extract.")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
