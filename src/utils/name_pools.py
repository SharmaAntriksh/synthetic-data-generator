# src/utils/name_pools.py
from __future__ import annotations

import os
import re
import unicodedata

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np


# ======================================================================================
# Public API (high-level)
# ======================================================================================

REGION_US = "US"
REGION_EU = "EU"
REGION_IN = "IN"
REGION_AS = "AS"

GENDER_MALE = "Male"
GENDER_FEMALE = "Female"


@dataclass(frozen=True)
class RegionNamePool:
    """
    Regional pool of names.
    - First names are gendered.
    - Last names are typically not gendered, but we support male/female for future cases.
    """
    male_first: np.ndarray
    female_first: np.ndarray
    male_last: np.ndarray
    female_last: np.ndarray

    def any_first(self) -> np.ndarray:
        if self.male_first.size and self.female_first.size:
            return np.concatenate([self.male_first, self.female_first])
        return self.male_first if self.male_first.size else self.female_first

    def any_last(self) -> np.ndarray:
        # If male/female are identical, just return one for efficiency.
        if self.male_last.size and self.female_last.size:
            if self.male_last.shape == self.female_last.shape and np.array_equal(self.male_last, self.female_last):
                return self.male_last
            return np.concatenate([self.male_last, self.female_last])
        return self.male_last if self.male_last.size else self.female_last


@dataclass(frozen=True)
class PeopleNamePools:
    """
    Container for multi-region person-name pools.

    Typical usage:
      pools = load_people_pools(folder)
      first, last, middle = assign_person_names(CustomerKey, Region, Gender, IsOrg, pools, seed)
    """
    folder: str
    regions: Dict[str, RegionNamePool]

    def region(self, code: str) -> RegionNamePool:
        rc = (code or "").upper()
        if rc in self.regions:
            return self.regions[rc]
        # fallback priority
        for fallback in (REGION_US, REGION_EU, REGION_IN, REGION_AS):
            if fallback in self.regions:
                return self.regions[fallback]
        raise KeyError(f"No regions loaded in PeopleNamePools (asked for {code!r}).")


# ======================================================================================
# Folder resolution (configs)
# ======================================================================================

def resolve_people_folder(
    cfg: Mapping[str, Any],
    *,
    per_module_folder: Optional[str] = None,
    default_folder: str = "./data/name_pools/people",
    legacy_folders: Iterable[str] = (),
) -> str:
    """
    Resolve a names folder consistently across generators.

    Lookup order:
      1) per_module_folder argument (e.g., customers.names_folder / employees.names_folder)
      2) cfg["names"]["people_folder"] (shared)
      3) default_folder
      4) legacy_folders (first existing)
    """
    if per_module_folder:
        return str(Path(per_module_folder))

    names_block = cfg.get("names") if isinstance(cfg, Mapping) else None
    if isinstance(names_block, Mapping):
        pf = names_block.get("people_folder")
        if pf:
            p = Path(pf)
            if not p.exists():
                raise FileNotFoundError(f"names.people_folder does not exist: {p}")
            return str(p)

    # If you want to hard-require the new config, fail here:
    raise ValueError("Missing config: names.people_folder (expected a valid folder path)")

# --- Org name pool -----------------------------------------------------

_slug_re = re.compile(r"[^a-z0-9\-]+")

def resolve_org_names_file(
    cfg: Mapping[str, Any],
    *,
    default_relpath: str = "org/org_names.csv",
) -> str:
    """
    Resolve org name list file path.

    Priority:
      1) cfg["names"]["org_names_file"] (explicit)
      2) sibling of people_folder: <parent_of_people_folder>/org/org_names.csv
    """
    names_block = cfg.get("names") if isinstance(cfg, Mapping) else None
    if isinstance(names_block, Mapping):
        explicit = names_block.get("org_names_file")
        if explicit:
            p = Path(explicit)
            if not p.exists():
                raise FileNotFoundError(f"names.org_names_file does not exist: {p}")
            return str(p)

        pf = names_block.get("people_folder")
        if pf:
            p = Path(pf)
            org_path = p.parent / default_relpath
            if not org_path.exists():
                raise FileNotFoundError(f"Expected org names at: {org_path} (derived from names.people_folder)")
            return str(org_path)

    raise ValueError("Missing config: names.people_folder (needed to derive org/org_names.csv), or set names.org_names_file.")


@lru_cache(maxsize=64)
def load_org_names(org_names_file: str) -> np.ndarray:
    """
    Load org names (one per line). Cached.
    """
    return load_list(org_names_file)


def slugify_domain_label(s: str) -> str:
    """
    'Northstar Logistics Ltd' -> 'northstarlogisticsltd'
    Keep [a-z0-9-], drop everything else.
    """
    t = normalize_name_ascii(s)  # already ASCII + title-cased tokens
    t = t.lower().replace(" ", "")
    t = _slug_re.sub("", t)
    return t[:63] if len(t) > 63 else t  # keep reasonable domain label length


def assign_org_names(
    *,
    keys: np.ndarray,
    is_org: np.ndarray,
    org_pool: np.ndarray,
    seed: int,
    salt: int = 424242,
) -> np.ndarray:
    """
    Deterministic org-name assignment by key (stable, low-clumping).
    Returns object array length N with None for non-org rows.
    """
    keys_u64 = np.asarray(keys, dtype=np.uint64)
    is_org = np.asarray(is_org, dtype=bool)

    out = np.empty(keys_u64.size, dtype=object)
    out[~is_org] = None

    if np.any(is_org):
        h = hash_u64(keys_u64, int(seed), int(salt))
        out[is_org] = pick_masked(np.asarray(org_pool, dtype=object), h, is_org, add_salt=7)
    return out

# ======================================================================================
# File reading + normalization
# ======================================================================================

_WS_RE = re.compile(r"\s+")
_ALLOWED_RE = re.compile(r"[^A-Za-z \-']")


def normalize_name_ascii(name: str, *, keep_apostrophe: bool = False) -> str:
    """
    Normalize to ASCII, remove punctuation, title-case.
    Keep hyphen; optionally keep apostrophe.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = _WS_RE.sub(" ", s).strip()
    if not keep_apostrophe:
        s = s.replace("'", "")
    s = _ALLOWED_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    # title-case each token, preserving hyphens
    tokens = []
    for tok in s.split(" "):
        parts = []
        for p in tok.split("-"):
            p = p.strip()
            parts.append((p[:1].upper() + p[1:].lower()) if p else "")
        tokens.append("-".join(parts))
    return " ".join(tokens).strip()


@lru_cache(maxsize=256)
def load_list(path: str, *, normalize: bool = True) -> np.ndarray:
    """
    Load a 1-column CSV / newline list: one name per line, no header.
    Caches by path string for the life of the process.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)

    # read lines; tolerate BOM
    raw = p.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    out: list[str] = []
    seen: set[str] = set()

    for line in raw:
        s = (line or "").strip()
        if not s:
            continue
        # allow accidental commas/quotes from CSV copy/paste
        s = s.strip().strip('"').strip("'")
        if "," in s and len(s.split(",")) > 1:
            # keep first column if someone pasted "Name,Other"
            s = s.split(",", 1)[0].strip()
        if normalize:
            s = normalize_name_ascii(s)
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)

    if not out:
        raise ValueError(f"Name list is empty after normalization: {path}")
    return np.asarray(out, dtype=object)


def _load_pool(
    folder: str,
    candidates: list[str],
    *,
    required: bool,
    label: str,
) -> np.ndarray:
    """
    Load the first existing and non-empty file among candidates.
    """
    base = Path(folder)
    tried = []
    for fname in candidates:
        tried.append(fname)
        p = base / fname
        if p.is_file():
            arr = load_list(str(p))
            if arr.size > 0:
                return arr
    if required:
        raise FileNotFoundError(f"Missing/empty name list for {label}. Tried: {tried} in folder={folder!r}")
    return np.asarray([], dtype=object)


# ======================================================================================
# Pool loader (backward compatible)
# ======================================================================================

def load_people_pools(
    folder: str,
    *,
    enable_asia: bool = True,
    legacy_support: bool = True,
) -> PeopleNamePools:
    """
    Load people name pools for US/EU/IN/(AS).

    New filenames:
      us_male_first.csv, us_female_first.csv, us_last.csv
      eu_male_first.csv, eu_female_first.csv, eu_last.csv
      india_male_first.csv, india_female_first.csv, india_last.csv
      asia_male_first.csv, asia_female_first.csv, asia_last.csv

    Optional gendered last names (supported if you add them later):
      {region}_male_last.csv, {region}_female_last.csv

    Legacy fallbacks:
      us_surnames.csv (for US last)
      eu_first.csv (for EU first if gendered files missing)
      india_first.csv (for IN first if gendered files missing)
      eu_last.csv / india_last.csv already match
    """
    folder = str(Path(folder))
    regions: Dict[str, RegionNamePool] = {}

    # US
    regions[REGION_US] = RegionNamePool(
        male_first=_load_pool(folder, ["us_male_first.csv"], required=True, label="US male_first"),
        female_first=_load_pool(folder, ["us_female_first.csv"], required=True, label="US female_first"),
        male_last=_load_pool(
            folder,
            ["us_male_last.csv", "us_last.csv"] + (["us_surnames.csv"] if legacy_support else []),
            required=True,
            label="US male_last",
        ),
        female_last=_load_pool(
            folder,
            ["us_female_last.csv", "us_last.csv"] + (["us_surnames.csv"] if legacy_support else []),
            required=True,
            label="US female_last",
        ),
    )

    # EU
    eu_male_first = _load_pool(
        folder,
        ["eu_male_first.csv"] + (["eu_first.csv"] if legacy_support else []),
        required=True,
        label="EU male_first",
    )
    eu_female_first = _load_pool(
        folder,
        ["eu_female_first.csv"] + (["eu_first.csv"] if legacy_support else []),
        required=True,
        label="EU female_first",
    )
    regions[REGION_EU] = RegionNamePool(
        male_first=eu_male_first,
        female_first=eu_female_first,
        male_last=_load_pool(folder, ["eu_male_last.csv", "eu_last.csv"], required=True, label="EU male_last"),
        female_last=_load_pool(folder, ["eu_female_last.csv", "eu_last.csv"], required=True, label="EU female_last"),
    )

    # IN
    in_male_first = _load_pool(
        folder,
        ["india_male_first.csv"] + (["india_first.csv"] if legacy_support else []),
        required=True,
        label="IN male_first",
    )
    in_female_first = _load_pool(
        folder,
        ["india_female_first.csv"] + (["india_first.csv"] if legacy_support else []),
        required=True,
        label="IN female_first",
    )
    regions[REGION_IN] = RegionNamePool(
        male_first=in_male_first,
        female_first=in_female_first,
        male_last=_load_pool(folder, ["india_male_last.csv", "india_last.csv"], required=True, label="IN male_last"),
        female_last=_load_pool(folder, ["india_female_last.csv", "india_last.csv"], required=True, label="IN female_last"),
    )

    # AS (optional)
    if enable_asia:
        # If you enable Asia but files are missing, this will raise (good: fail fast).
        regions[REGION_AS] = RegionNamePool(
            male_first=_load_pool(folder, ["asia_male_first.csv"], required=True, label="AS male_first"),
            female_first=_load_pool(folder, ["asia_female_first.csv"], required=True, label="AS female_first"),
            male_last=_load_pool(folder, ["asia_male_last.csv", "asia_last.csv"], required=True, label="AS male_last"),
            female_last=_load_pool(folder, ["asia_female_last.csv", "asia_last.csv"], required=True, label="AS female_last"),
        )

    return PeopleNamePools(folder=folder, regions=regions)


# ======================================================================================
# Deterministic hashing + selection (generic utilities)
# ======================================================================================

def hash_u64(keys: np.ndarray, seed: int, salt: int) -> np.ndarray:
    """
    Deterministic 64-bit hash for stable name assignment.

    Works well for:
      - CustomerKey
      - EmployeeKey
      - StoreKey (e.g., StoreManagerName)
    """
    h = np.asarray(keys, dtype=np.uint64)
    h = h * np.uint64(2654435761) + np.uint64(seed) * np.uint64(1013904223) + np.uint64(salt)
    h ^= (h >> np.uint64(16))
    h *= np.uint64(2246822519)
    h ^= (h >> np.uint64(13))
    return h


def pick_from_pool(pool: np.ndarray, h: np.ndarray, *, add_salt: int = 0) -> np.ndarray:
    """
    Vectorized selection: returns array same shape as h.
    """
    pool = np.asarray(pool, dtype=object)
    if pool.size == 0:
        raise ValueError("Pool is empty")
    idx = ((h + np.uint64(add_salt)) % np.uint64(pool.size)).astype(np.int64)
    return pool[idx]


def pick_masked(pool: np.ndarray, h: np.ndarray, mask: np.ndarray, *, add_salt: int = 0) -> np.ndarray:
    """
    Masked selection: returns selected values of length mask.sum().
    """
    pool = np.asarray(pool, dtype=object)
    if pool.size == 0:
        raise ValueError("Pool is empty")
    idx = ((h[mask] + np.uint64(add_salt)) % np.uint64(pool.size)).astype(np.int64)
    return pool[idx]


def middle_initial(keys: np.ndarray, seed: int, *, salt: int = 911) -> np.ndarray:
    """
    Deterministic middle initial, returned as e.g. 'K.' (object dtype).
    """
    letters = np.asarray(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), dtype=object)
    h = hash_u64(np.asarray(keys, dtype=np.uint64), int(seed), int(salt))
    mid = letters[(h % np.uint64(26)).astype(np.int64)]
    return (mid.astype(object).astype(str) + ".").astype(object)


# ======================================================================================
# High-level assignment helpers (use directly in dimensions)
# ======================================================================================

def assign_person_names(
    *,
    keys: np.ndarray,
    region: np.ndarray,
    gender: np.ndarray,
    is_org: Optional[np.ndarray],
    pools: PeopleNamePools,
    seed: int,
    include_middle: bool = False,
    default_region: str = REGION_US,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Assign (FirstName, LastName, MiddleName?) based on:
      - keys: stable IDs (CustomerKey/EmployeeKey/etc)
      - region: 'US'/'EU'/'IN'/'AS' (or anything; falls back)
      - gender: 'Male'/'Female' (others fall back to any_first)
      - is_org: if True -> name fields are None/empty (you decide downstream)

    Returns arrays dtype=object.
    """
    keys_u64 = np.asarray(keys, dtype=np.uint64)
    region = np.asarray(region, dtype=object)
    gender = np.asarray(gender, dtype=object)
    N = int(keys_u64.size)

    if is_org is None:
        is_org = np.zeros(N, dtype=bool)
    else:
        is_org = np.asarray(is_org, dtype=bool)

    first = np.empty(N, dtype=object)
    last = np.empty(N, dtype=object)
    mid: Optional[np.ndarray] = None

    h_first = hash_u64(keys_u64, int(seed), 11)
    h_last = hash_u64(keys_u64, int(seed), 29)

    # Normalize region codes: upper, fallback to default
    def _norm_region(x: Any) -> str:
        s = str(x).upper() if x is not None else ""
        return s if s in pools.regions else default_region

    region_norm = np.asarray([_norm_region(r) for r in region], dtype=object)

    for rc, salt_base in ((REGION_IN, 1000), (REGION_US, 2000), (REGION_EU, 3000), (REGION_AS, 4000)):
        if rc not in pools.regions:
            continue
        mask_r = (region_norm == rc) & (~is_org)
        if not np.any(mask_r):
            continue

        rp = pools.regions[rc]

        male_mask = mask_r & (gender == GENDER_MALE)
        female_mask = mask_r & (gender == GENDER_FEMALE)
        other_mask = mask_r & (~(gender == GENDER_MALE) & ~(gender == GENDER_FEMALE))

        # Last names (mostly not gendered, but supported)
        if np.any(male_mask):
            last[male_mask] = pick_masked(rp.male_last, h_last, male_mask, add_salt=salt_base + 1)
        if np.any(female_mask):
            last[female_mask] = pick_masked(rp.female_last, h_last, female_mask, add_salt=salt_base + 2)
        if np.any(other_mask):
            last[other_mask] = pick_masked(rp.any_last(), h_last, other_mask, add_salt=salt_base + 3)

        # First names (gendered)
        if np.any(male_mask):
            first[male_mask] = pick_masked(rp.male_first, h_first, male_mask, add_salt=salt_base + 11)
        if np.any(female_mask):
            first[female_mask] = pick_masked(rp.female_first, h_first, female_mask, add_salt=salt_base + 12)
        if np.any(other_mask):
            first[other_mask] = pick_masked(rp.any_first(), h_first, other_mask, add_salt=salt_base + 13)

    # Org rows => None (caller can turn into blanks)
    first[is_org] = None
    last[is_org] = None

    if include_middle:
        mid = middle_initial(keys_u64, int(seed))
        mid[is_org] = None

    return first, last, mid


__all__ = [
    "REGION_US",
    "REGION_EU",
    "REGION_IN",
    "REGION_AS",
    "GENDER_MALE",
    "GENDER_FEMALE",
    "RegionNamePool",
    "PeopleNamePools",
    "resolve_people_folder",
    "normalize_name_ascii",
    "load_list",
    "load_people_pools",
    "hash_u64",
    "pick_from_pool",
    "pick_masked",
    "middle_initial",
    "assign_person_names",
    "resolve_org_names_file",
    "load_org_names",
    "slugify_domain_label",
    "assign_org_names",
]