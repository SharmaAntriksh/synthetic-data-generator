"""Post-generation data quality report.

Reads the final output folder, checks referential integrity,
distribution sanity, and null/duplicate checks, then writes
an HTML report alongside the data.
"""
from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import info, stage


# ============================================================================
# Check result model
# ============================================================================

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: str = ""
    category: str = "General"


@dataclass
class DistributionSummary:
    column: str
    table: str
    min_val: Any = None
    max_val: Any = None
    mean_val: Any = None
    median_val: Any = None
    std_val: Any = None
    top_values: List[tuple] = field(default_factory=list)


@dataclass
class QualityReport:
    checks: List[CheckResult] = field(default_factory=list)
    distributions: List[DistributionSummary] = field(default_factory=list)
    table_row_counts: Dict[str, int] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)


# ============================================================================
# File loaders (format-aware)
# ============================================================================

def _read_table(path: Path, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Read a table from parquet, CSV dir, or delta — returns None if missing."""
    if path.is_file() and path.suffix == ".parquet":
        try:
            return pd.read_parquet(path, columns=columns)
        except Exception:
            return None

    # CSV: could be a single file or a directory of chunks
    if path.is_file() and path.suffix == ".csv":
        try:
            return pd.read_csv(path, usecols=columns)
        except Exception:
            return None

    if path.is_dir():
        # Delta table?
        if (path / "_delta_log").is_dir():
            try:
                from deltalake import DeltaTable
                dt = DeltaTable(str(path))
                df = dt.to_pandas()
                if columns:
                    available = [c for c in columns if c in df.columns]
                    df = df[available]
                return df
            except Exception:
                return None

        # CSV chunk directory
        csvs = sorted(path.glob("*.csv"))
        if csvs:
            try:
                frames = [pd.read_csv(f, usecols=columns) for f in csvs]
                return pd.concat(frames, ignore_index=True)
            except Exception:
                return None

        # Parquet files in directory
        pqs = sorted(path.glob("*.parquet"))
        if pqs:
            try:
                frames = [pd.read_parquet(f, columns=columns) for f in pqs]
                return pd.concat(frames, ignore_index=True)
            except Exception:
                return None

    return None


def _find_table(folder: Path, name: str) -> Optional[Path]:
    """Find a table by name in dims or facts folder."""
    # Try common naming: snake_case parquet, PascalCase, lowercase
    candidates = [
        folder / f"{name}.parquet",
        folder / f"{name}.csv",
        folder / name,  # directory (delta or csv chunks)
    ]
    # Also try snake_case conversion
    snake = _to_snake(name)
    if snake != name:
        candidates.extend([
            folder / f"{snake}.parquet",
            folder / f"{snake}.csv",
            folder / snake,
        ])

    for c in candidates:
        if c.exists():
            return c
    return None


def _to_snake(name: str) -> str:
    """PascalCase to snake_case."""
    import re
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    return s


# ============================================================================
# Check runners
# ============================================================================

def _check_referential_integrity(
    report: QualityReport,
    dims_folder: Path,
    facts_folder: Path,
) -> None:
    """Check that all FK values in fact tables exist in dimension tables."""

    # FK relationships: (fact_table, fk_column, dim_table, pk_column)
    fk_checks = [
        ("sales", "CustomerKey", "customers", "CustomerKey"),
        ("sales", "ProductKey", "products", "ProductKey"),
        ("sales", "StoreKey", "stores", "StoreKey"),
        ("sales", "PromotionKey", "promotions", "PromotionKey"),
        ("sales", "CurrencyKey", "currency", "CurrencyKey"),
        ("sales", "SalesPersonEmployeeKey", "employees", "EmployeeKey"),
        ("sales", "SalesChannelKey", "sales_channels", "SalesChannelKey"),
        ("inventory_snapshot", "ProductKey", "products", "ProductKey"),
        ("inventory_snapshot", "StoreKey", "stores", "StoreKey"),
    ]

    for fact_name, fk_col, dim_name, pk_col in fk_checks:
        fact_path = _find_table(facts_folder, fact_name)
        dim_path = _find_table(dims_folder, dim_name)

        if fact_path is None:
            continue  # Table not generated, skip

        if dim_path is None:
            report.checks.append(CheckResult(
                name=f"FK: {fact_name}.{fk_col} → {dim_name}.{pk_col}",
                passed=False,
                message=f"Dimension table '{dim_name}' not found",
                category="Referential Integrity",
            ))
            continue

        fact_df = _read_table(fact_path, columns=[fk_col])
        dim_df = _read_table(dim_path, columns=[pk_col])

        if fact_df is None or dim_df is None:
            continue

        if fk_col not in fact_df.columns or pk_col not in dim_df.columns:
            continue

        fact_keys = set(fact_df[fk_col].dropna().unique())
        dim_keys = set(dim_df[pk_col].dropna().unique())
        orphans = fact_keys - dim_keys

        if orphans:
            sample = sorted(orphans)[:10]
            report.checks.append(CheckResult(
                name=f"FK: {fact_name}.{fk_col} → {dim_name}.{pk_col}",
                passed=False,
                message=f"{len(orphans)} orphan key(s) in {fact_name}.{fk_col}",
                details=f"Sample orphans: {sample}",
                category="Referential Integrity",
            ))
        else:
            report.checks.append(CheckResult(
                name=f"FK: {fact_name}.{fk_col} → {dim_name}.{pk_col}",
                passed=True,
                message=f"All {len(fact_keys)} keys found in {dim_name}",
                category="Referential Integrity",
            ))


def _check_nulls_and_duplicates(
    report: QualityReport,
    dims_folder: Path,
    facts_folder: Path,
) -> None:
    """Check for null PKs and duplicate PKs in key tables."""

    pk_checks = [
        ("customers", "CustomerKey", dims_folder),
        ("products", "ProductKey", dims_folder),
        ("stores", "StoreKey", dims_folder),
        ("promotions", "PromotionKey", dims_folder),
        ("employees", "EmployeeKey", dims_folder),
        ("dates", "DateKey", dims_folder),
        ("geography", "GeographyKey", dims_folder),
        ("currency", "CurrencyKey", dims_folder),
    ]

    for table_name, pk_col, folder in pk_checks:
        path = _find_table(folder, table_name)
        if path is None:
            continue

        df = _read_table(path, columns=[pk_col])
        if df is None or pk_col not in df.columns:
            continue

        # Null check
        null_count = int(df[pk_col].isna().sum())
        report.checks.append(CheckResult(
            name=f"Null PK: {table_name}.{pk_col}",
            passed=null_count == 0,
            message=f"{null_count} null(s)" if null_count > 0 else "No nulls",
            category="Null / Duplicate",
        ))

        # Duplicate check
        dup_count = int(df[pk_col].duplicated().sum())
        report.checks.append(CheckResult(
            name=f"Duplicate PK: {table_name}.{pk_col}",
            passed=dup_count == 0,
            message=f"{dup_count} duplicate(s)" if dup_count > 0 else "No duplicates",
            category="Null / Duplicate",
        ))


def _check_distributions(
    report: QualityReport,
    dims_folder: Path,
    facts_folder: Path,
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Check that key distributions look reasonable."""

    # Sales: quantity distribution
    sales_path = _find_table(facts_folder, "sales")
    if sales_path is not None:
        sales_df = _read_table(sales_path, columns=["Quantity", "NetPrice", "OrderDate", "DiscountAmount"])
        if sales_df is not None:
            if "Quantity" in sales_df.columns:
                qty = sales_df["Quantity"]
                report.distributions.append(DistributionSummary(
                    column="Quantity", table="Sales",
                    min_val=int(qty.min()), max_val=int(qty.max()),
                    mean_val=round(float(qty.mean()), 2),
                    median_val=round(float(qty.median()), 2),
                    std_val=round(float(qty.std()), 2),
                ))
                # Sanity: avg qty should be reasonable (0.5 - 10)
                avg_qty = float(qty.mean())
                report.checks.append(CheckResult(
                    name="Avg basket size (Quantity)",
                    passed=0.5 <= avg_qty <= 10,
                    message=f"Mean={avg_qty:.2f}",
                    details="Expected between 0.5 and 10 based on typical Poisson config",
                    category="Distribution",
                ))

            if "NetPrice" in sales_df.columns:
                price = sales_df["NetPrice"]
                report.distributions.append(DistributionSummary(
                    column="NetPrice", table="Sales",
                    min_val=round(float(price.min()), 2),
                    max_val=round(float(price.max()), 2),
                    mean_val=round(float(price.mean()), 2),
                    median_val=round(float(price.median()), 2),
                    std_val=round(float(price.std()), 2),
                ))
                # Sanity: no negative net prices
                neg_count = int((price < 0).sum())
                report.checks.append(CheckResult(
                    name="Negative NetPrice",
                    passed=neg_count == 0,
                    message=f"{neg_count} negative price(s)" if neg_count > 0 else "All prices non-negative",
                    category="Distribution",
                ))

            if "OrderDate" in sales_df.columns:
                dates = pd.to_datetime(sales_df["OrderDate"])
                report.checks.append(CheckResult(
                    name="Sales date range",
                    passed=True,
                    message=f"{dates.min().date()} to {dates.max().date()}",
                    category="Distribution",
                ))

                # Monthly distribution
                monthly = dates.dt.to_period("M").value_counts().sort_index()
                top_months = [(str(k), int(v)) for k, v in monthly.head(3).items()]
                report.distributions.append(DistributionSummary(
                    column="OrderDate (monthly)", table="Sales",
                    min_val=int(monthly.min()), max_val=int(monthly.max()),
                    mean_val=round(float(monthly.mean()), 0),
                    top_values=top_months,
                ))

    # Returns: check return rate if sales return exists
    returns_path = _find_table(facts_folder, "sales_return")
    if returns_path is not None and sales_path is not None:
        returns_df = _read_table(returns_path)
        sales_df_count = _read_table(sales_path, columns=["Quantity"])
        if returns_df is not None and sales_df_count is not None:
            return_rate = len(returns_df) / max(len(sales_df_count), 1)
            report.checks.append(CheckResult(
                name="Return rate",
                passed=True,
                message=f"{return_rate:.2%} ({len(returns_df)} returns / {len(sales_df_count)} sales)",
                category="Distribution",
            ))

    # Customers per geography
    cust_path = _find_table(dims_folder, "customers")
    if cust_path is not None:
        cust_df = _read_table(cust_path, columns=["GeographyKey"])
        if cust_df is not None and "GeographyKey" in cust_df.columns:
            geo_dist = cust_df["GeographyKey"].value_counts()
            report.distributions.append(DistributionSummary(
                column="GeographyKey", table="Customers",
                min_val=int(geo_dist.min()), max_val=int(geo_dist.max()),
                mean_val=round(float(geo_dist.mean()), 0),
            ))


def _collect_row_counts(
    report: QualityReport,
    dims_folder: Path,
    facts_folder: Path,
) -> None:
    """Count rows in all output tables."""
    for folder, label in [(dims_folder, "dim"), (facts_folder, "fact")]:
        if not folder.exists():
            continue
        for item in sorted(folder.iterdir()):
            df = _read_table(item)
            if df is not None:
                name = item.stem if item.is_file() else item.name
                report.table_row_counts[f"{label}/{name}"] = len(df)


# ============================================================================
# HTML renderer
# ============================================================================

def _render_html(report: QualityReport) -> str:
    """Render the report as a standalone HTML page."""
    check_rows = []
    for c in report.checks:
        status = "PASS" if c.passed else "FAIL"
        css = "pass" if c.passed else "fail"
        detail = f'<div class="detail">{html.escape(c.details)}</div>' if c.details else ""
        check_rows.append(
            f'<tr class="{css}">'
            f'<td class="status">{status}</td>'
            f'<td>{html.escape(c.category)}</td>'
            f'<td>{html.escape(c.name)}</td>'
            f'<td>{html.escape(c.message)}{detail}</td>'
            f'</tr>'
        )

    dist_rows = []
    for d in report.distributions:
        top = ", ".join(f"{k}: {v}" for k, v in d.top_values) if d.top_values else ""
        dist_rows.append(
            f'<tr>'
            f'<td>{html.escape(d.table)}</td>'
            f'<td>{html.escape(d.column)}</td>'
            f'<td>{d.min_val}</td>'
            f'<td>{d.max_val}</td>'
            f'<td>{d.mean_val}</td>'
            f'<td>{d.median_val if d.median_val is not None else ""}</td>'
            f'<td>{d.std_val if d.std_val is not None else ""}</td>'
            f'<td>{html.escape(top)}</td>'
            f'</tr>'
        )

    count_rows = []
    for name, count in sorted(report.table_row_counts.items()):
        count_rows.append(f'<tr><td>{html.escape(name)}</td><td>{count:,}</td></tr>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Data Quality Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f5; color: #333; padding: 2rem; }}
  h1 {{ margin-bottom: 0.5rem; }}
  .summary {{ font-size: 1.1rem; margin-bottom: 1.5rem; color: #555; }}
  .summary .pass-count {{ color: #16a34a; font-weight: 600; }}
  .summary .fail-count {{ color: #dc2626; font-weight: 600; }}
  h2 {{ margin: 1.5rem 0 0.75rem; border-bottom: 2px solid #ddd; padding-bottom: 0.25rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; background: #fff;
           border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #f8f8f8; text-align: left; padding: 0.6rem 0.8rem; font-weight: 600;
       border-bottom: 2px solid #e5e5e5; font-size: 0.85rem; text-transform: uppercase;
       letter-spacing: 0.03em; }}
  td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid #eee; font-size: 0.9rem; }}
  tr:last-child td {{ border-bottom: none; }}
  tr.pass .status {{ color: #16a34a; font-weight: 700; }}
  tr.fail {{ background: #fef2f2; }}
  tr.fail .status {{ color: #dc2626; font-weight: 700; }}
  .detail {{ font-size: 0.8rem; color: #888; margin-top: 0.2rem; }}
  .elapsed {{ font-size: 0.85rem; color: #999; margin-top: 1rem; }}
</style>
</head>
<body>
<h1>Data Quality Report</h1>
<div class="summary">
  <span class="pass-count">{report.passed} passed</span> &middot;
  <span class="fail-count">{report.failed} failed</span> &middot;
  {report.total} total checks
</div>

<h2>Checks</h2>
<table>
<thead><tr><th>Status</th><th>Category</th><th>Check</th><th>Result</th></tr></thead>
<tbody>
{"".join(check_rows)}
</tbody>
</table>

<h2>Row Counts</h2>
<table>
<thead><tr><th>Table</th><th>Rows</th></tr></thead>
<tbody>
{"".join(count_rows)}
</tbody>
</table>

<h2>Distribution Summaries</h2>
<table>
<thead><tr><th>Table</th><th>Column</th><th>Min</th><th>Max</th><th>Mean</th><th>Median</th><th>StdDev</th><th>Top Values</th></tr></thead>
<tbody>
{"".join(dist_rows)}
</tbody>
</table>

<div class="elapsed">Report generated in {report.elapsed_sec:.2f}s</div>
</body>
</html>"""


# ============================================================================
# Public API
# ============================================================================

def generate_quality_report(
    final_folder: Path,
    cfg: Optional[Dict[str, Any]] = None,
) -> Path:
    """Run all quality checks on the output in *final_folder* and write an HTML report.

    Returns the path to the generated HTML file.
    """
    with stage("Data Quality Report"):
        t0 = time.time()
        report = QualityReport()

        dims_folder = final_folder / "dimensions"
        facts_folder = final_folder / "facts"

        if not dims_folder.exists() and not facts_folder.exists():
            info(f"No dimensions/ or facts/ folder found in {final_folder}, skipping report.")
            return final_folder / "data_quality_report.html"

        info("Running referential integrity checks...")
        _check_referential_integrity(report, dims_folder, facts_folder)

        info("Running null / duplicate checks...")
        _check_nulls_and_duplicates(report, dims_folder, facts_folder)

        info("Checking distributions...")
        _check_distributions(report, dims_folder, facts_folder, cfg)

        info("Counting rows...")
        _collect_row_counts(report, dims_folder, facts_folder)

        report.elapsed_sec = time.time() - t0

        out_path = final_folder / "data_quality_report.html"
        out_path.write_text(_render_html(report), encoding="utf-8")
        info(f"Quality report: {report.passed} passed, {report.failed} failed ({report.total} checks)")
        info(f"Report written - {out_path.name}")

        return out_path
