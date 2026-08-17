from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetCase:
    name: str
    directory: str
    max_history: int


def load_table_config(path: Path) -> dict:
    return json.loads(path.read_text())
