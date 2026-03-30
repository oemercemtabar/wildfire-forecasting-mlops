import json
from pathlib import Path
from typing import Any

import joblib


def ensure_directories(paths: list[Path]) -> None:
    """Ensure that the directories for the given path exist."""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as a JSON file."""
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and return it as a dictionary."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_joblib(obj: Any, path: str | Path) -> None:
    """Save data using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: str | Path) -> Any:
    """Load data using joblib."""
    path = Path(path)
    return joblib.load(path)
