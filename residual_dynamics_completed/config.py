"""Fixed approved protocol; paths resolve only inside this package."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ("nominal", "profile_only", "friction_only", "rbf_pv", "mlp", "structured")

def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    return value

def configuration():
    return json.loads((ROOT / "configs/comparison.json").read_text())

def config_hash():
    return hashlib.sha256(json.dumps(configuration(), sort_keys=True).encode()).hexdigest()

def load_config():
    cfg = namespace(configuration())
    cfg.paths = SimpleNamespace(processed_dir=ROOT / "data/processed")
    return cfg
