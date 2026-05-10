"""Load repo-root `.env` into the process environment for integration tests.

Does not override variables already set (e.g. CI secrets).
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / ".env"
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def pytest_configure() -> None:
    _load_dotenv()
