from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMID_DIR = DATA_DIR / "intermid"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"


CONFGIS_DIR = ROOT_DIR / "configs"
LOGS_DIR = ROOT_DIR / "logs"
SRC_DIR = ROOT_DIR / "src"

