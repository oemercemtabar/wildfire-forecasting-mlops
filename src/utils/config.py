from pathlib import Path
from typing import Any 

import yaml

def load_yaml(path: str | Path) -> dict[str, Any]:
    
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML config file not found: {path}")
    
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object at root of {path}, got {type(data).__name__}")
    
    return data