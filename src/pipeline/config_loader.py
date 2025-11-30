import copy
from datetime import datetime

def load_config(cfg):
    """
    Returns a dict of fully-resolved module configs:
        resolved["sales"]
        resolved["promotions"]
        resolved["customers"]
        etc.
    each with final: seed, dates, paths, and local fields.
    """
    resolved = {}
    defaults = cfg.get("defaults", {})

    for section_name, section in cfg.items():
        if section_name == "defaults":
            continue
        resolved[section_name] = resolve_section(cfg, section_name, defaults)

    return resolved


def resolve_section(cfg, section_name, defaults):
    """
    Merge: defaults → module_section → override
    + special logic for promotions.date_ranges
    """
    section = copy.deepcopy(cfg.get(section_name, {}))
    override = section.pop("override", {})

    # -----------------------------------------
    # Base structure
    # -----------------------------------------
    out = {
        "seed": defaults.get("seed"),
        "dates": copy.deepcopy(defaults.get("dates", {})),
        "paths": copy.deepcopy(defaults.get("paths", {}))
    }

    # -----------------------------------------
    # Merge module-level fields (non override)
    # -----------------------------------------
    for key, val in section.items():
        if key not in ("dates", "paths"):  # reserved keys
            out[key] = val

    # -----------------------------------------
    # SPECIAL: PROMOTIONS DATE RANGES
    # -----------------------------------------
    if section_name == "promotions":
        date_ranges = section.get("date_ranges", [])
        if date_ranges:  # user provided custom promo windows
            out["date_ranges"] = date_ranges
        else:
            # Convert global start/end → single range
            out["date_ranges"] = [{
                "start": out["dates"]["start"],
                "end": out["dates"]["end"]
            }]

    # -----------------------------------------
    # Apply override: dates
    # -----------------------------------------
    if "dates" in override and isinstance(override["dates"], dict):
        out["dates"] = {
            **out["dates"],
            **override["dates"]
        }

    # -----------------------------------------
    # Apply override: seed
    # -----------------------------------------
    if override.get("seed") is not None:
        out["seed"] = override["seed"]

    # -----------------------------------------
    # Apply override: paths
    # -----------------------------------------
    if "paths" in override and isinstance(override["paths"], dict):
        out["paths"] = {
            **out["paths"],
            **override["paths"]
        }

    return out
