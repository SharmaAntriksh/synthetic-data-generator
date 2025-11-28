import json
import hashlib
from pathlib import Path

# Folder to store version metadata
VERSION_DIR = Path("data/versioning")
VERSION_DIR.mkdir(parents=True, exist_ok=True)


def compute_hash(obj):
    """Compute deterministic hash of a Python object."""
    data = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def version_file_for(name: str):
    """Return the full path to the version file for this dimension."""
    return VERSION_DIR / f"{name}.version.json"


def load_version(name: str):
    vf = version_file_for(name)
    if not vf.exists():
        return None
    try:
        return json.loads(vf.read_text())
    except:
        return None


def save_version(name: str, cfg_section):
    vf = version_file_for(name)
    data = {"config_hash": compute_hash(cfg_section)}
    vf.write_text(json.dumps(data, indent=2))


def should_regenerate(name: str, cfg_section, parquet_path):
    """
    True if this dimension must be regenerated.
    Condition:
    - parquet is missing OR
    - version file missing OR
    - config hash changed
    """
    current_hash = compute_hash(cfg_section)
    old = load_version(name)

    if (not parquet_path.exists() or
        old is None or
        old.get("config_hash") != current_hash):
        return True

    return False
