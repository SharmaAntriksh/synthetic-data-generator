"""Runner for the dates dimension — config parsing, versioning, parquet write."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.exceptions import ConfigError, DimensionError
from src.utils import info, warn, skip, stage
from src.utils.config_helpers import int_or as _int_or, bool_or as _bool_or, as_dict
from src.utils.output_utils import write_parquet_with_date32
from src.versioning import should_regenerate, save_version

from .columns import get_date_rename_map, resolve_date_columns
from .generator import generate_date_table
from .helpers import _clamp_month, _normalize_override_dates, _require_start_end
from .weekly_fiscal import WeeklyFiscalConfig


def run_dates(cfg: Dict, parquet_folder: Path) -> None:
    out_path = parquet_folder / "dates.parquet"
    if not hasattr(cfg, "dates"):
        raise ConfigError("Missing required config section: 'dates'")

    dates_cfg = cfg.dates or {}

    defaults_dates = (
        getattr(cfg.defaults, "dates", None) if hasattr(cfg, "defaults") else None
    ) or (
        getattr(getattr(cfg, "_defaults", None), "dates", None)
    ) or {}

    version_cfg = as_dict(dates_cfg)
    version_cfg["global_dates"] = as_dict(defaults_dates) if defaults_dates else {}

    if not should_regenerate("dates", version_cfg, out_path):
        skip("Dates up-to-date")
        return

    override_dates = _normalize_override_dates(dates_cfg)

    raw_start = override_dates.get("start") or getattr(defaults_dates, "start", None)
    raw_end = override_dates.get("end") or getattr(defaults_dates, "end", None)
    raw_start_ts, raw_end_ts = _require_start_end(raw_start, raw_end)

    buffer_years = max(0, _int_or(dates_cfg.buffer_years, 1))
    start_date = pd.Timestamp(raw_start_ts.year - buffer_years, 1, 1)
    end_date = pd.Timestamp(raw_end_ts.year + buffer_years, 12, 31)
    info(f"Dates: {start_date.date()} to {end_date.date()} (±{buffer_years}yr buffer)")

    fiscal_start_month = _clamp_month(_int_or(dates_cfg.fiscal_start_month, 5))

    as_of_date = dates_cfg.as_of_date or str(raw_end_ts.date())

    compression = dates_cfg.parquet_compression
    compression_level = dates_cfg.parquet_compression_level
    force_date32 = _bool_or(dates_cfg.force_date32, True)

    include = dates_cfg.include or {}
    wf_block = getattr(include, "weekly_fiscal", None) or {}
    if isinstance(wf_block, bool):
        wf_block = {"enabled": wf_block}
    if isinstance(wf_block, Mapping):
        wf_cfg = WeeklyFiscalConfig(
            enabled=_bool_or(wf_block.get("enabled", True), True),
            first_day_of_week=_int_or(wf_block.get("first_day_of_week", 0), 0),
            weekly_type=str(wf_block.get("weekly_type", "Last")),
            quarter_week_type=str(wf_block.get("quarter_week_type", "445")),
            type_start_fiscal_year=_int_or(wf_block.get("type_start_fiscal_year", 1), 1),
        )
    else:
        # wf_block is a Pydantic model (WeeklyFiscalConfig from config_schema)
        wf_cfg = WeeklyFiscalConfig(
            enabled=_bool_or(getattr(wf_block, "enabled", True), True),
            first_day_of_week=_int_or(getattr(wf_block, "first_day_of_week", 0), 0),
            weekly_type=str(getattr(wf_block, "weekly_type", "Last")),
            quarter_week_type=str(getattr(wf_block, "quarter_week_type", "445")),
            type_start_fiscal_year=_int_or(getattr(wf_block, "type_start_fiscal_year", 1), 1),
        )

    with stage("Generating Dates"):
        df = generate_date_table(
            start_date,
            end_date,
            fiscal_start_month,
            as_of_date=as_of_date,
            weekly_cfg=wf_cfg,
        )

        rename_map = get_date_rename_map(dates_cfg)
        if rename_map:
            df = df.rename(columns=rename_map)

        cols = resolve_date_columns(dates_cfg)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise DimensionError(f"Dates: requested columns missing from generator: {missing}")

        df = df[cols]

        write_parquet_with_date32(
            df,
            out_path,
            compression=compression,
            compression_level=compression_level,
            force_date32=force_date32,
        )

    save_version("dates", version_cfg, out_path)
    info(f"Dates dimension written: {out_path.name}")
