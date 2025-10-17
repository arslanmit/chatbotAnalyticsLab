"""File system utilities for backups and persistence."""

from __future__ import annotations

import shutil
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def rotate_files(directory: Path, retention: int) -> None:
    files = sorted(directory.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for index, file in enumerate(files):
        if index >= retention:
            if file.is_dir():
                shutil.rmtree(file, ignore_errors=True)
            else:
                file.unlink(missing_ok=True)
