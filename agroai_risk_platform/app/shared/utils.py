import joblib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("agroai_risk")
logging.basicConfig(level=logging.INFO)


def load_artifact(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
