"""Specification loader for the edge analyzer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

__all__ = ["load_spec"]


def load_spec(path: str | Path) -> Dict[str, Any]:
    """Load a JSON specification file."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Spec file not found: {file_path}")
    with file_path.open("r", encoding="utf8") as handle:
        return json.load(handle)
