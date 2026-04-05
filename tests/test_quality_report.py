"""Tests for src.engine.quality_report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestQualityReport:
    def test_smoke_generates_html(self, tmp_path):
        from src.engine.quality_report import generate_quality_report

        dims = tmp_path / "dimensions"
        dims.mkdir()
        facts = tmp_path / "facts"
        facts.mkdir()

        customers = pd.DataFrame({
            "CustomerKey": np.arange(1, 6, dtype=np.int64),
            "CustomerName": [f"Cust {i}" for i in range(1, 6)],
        })
        customers.to_parquet(dims / "customers.parquet", index=False)

        products = pd.DataFrame({
            "ProductKey": np.arange(1, 4, dtype=np.int64),
            "ProductName": ["A", "B", "C"],
        })
        products.to_parquet(dims / "products.parquet", index=False)

        stores = pd.DataFrame({
            "StoreKey": np.arange(1, 3, dtype=np.int64),
            "StoreName": ["S1", "S2"],
        })
        stores.to_parquet(dims / "stores.parquet", index=False)

        dates = pd.DataFrame({
            "DateKey": pd.date_range("2024-01-01", periods=10),
        })
        dates.to_parquet(dims / "dates.parquet", index=False)

        sales = pd.DataFrame({
            "CustomerKey": np.array([1, 2, 3, 4, 5], dtype=np.int64),
            "ProductKey": np.array([1, 2, 3, 1, 2], dtype=np.int64),
            "StoreKey": np.array([1, 2, 1, 2, 1], dtype=np.int64),
            "OrderDate": pd.date_range("2024-01-01", periods=5),
            "Quantity": [1, 2, 1, 3, 1],
            "NetPrice": [10.0, 20.0, 30.0, 40.0, 50.0],
            "DiscountAmount": [0.0, 1.0, 0.0, 2.0, 0.0],
        })
        sales.to_parquet(facts / "sales.parquet", index=False)

        report_path = generate_quality_report(tmp_path)
        assert report_path.exists()
        assert report_path.suffix == ".html"
        html = report_path.read_text(encoding="utf-8")
        assert "<html" in html.lower()
        assert "customers" in html.lower()

    def test_html_contains_row_counts(self, tmp_path):
        """Row counts for each table should appear in the HTML report."""
        from src.engine.quality_report import generate_quality_report

        dims = tmp_path / "dimensions"
        dims.mkdir()
        facts = tmp_path / "facts"
        facts.mkdir()

        customers = pd.DataFrame({
            "CustomerKey": np.arange(1, 11, dtype=np.int64),
            "CustomerName": [f"Cust {i}" for i in range(1, 11)],
        })
        customers.to_parquet(dims / "customers.parquet", index=False)

        sales = pd.DataFrame({
            "CustomerKey": np.arange(1, 6, dtype=np.int64),
            "ProductKey": np.array([1, 2, 3, 1, 2], dtype=np.int64),
            "StoreKey": np.array([1, 2, 1, 2, 1], dtype=np.int64),
            "OrderDate": pd.date_range("2024-01-01", periods=5),
            "Quantity": [1, 2, 1, 3, 1],
            "NetPrice": [10.0, 20.0, 30.0, 40.0, 50.0],
            "DiscountAmount": [0.0, 1.0, 0.0, 2.0, 0.0],
        })
        sales.to_parquet(facts / "sales.parquet", index=False)

        report_path = generate_quality_report(tmp_path)
        html = report_path.read_text(encoding="utf-8")
        assert "10" in html  # 10 customers
        assert "sales" in html.lower()

    def test_html_structure(self, tmp_path):
        """Report should contain proper HTML structure with tables."""
        from src.engine.quality_report import generate_quality_report

        dims = tmp_path / "dimensions"
        dims.mkdir()
        pd.DataFrame({"CustomerKey": [1, 2]}).to_parquet(
            dims / "customers.parquet", index=False)

        report_path = generate_quality_report(tmp_path)
        html = report_path.read_text(encoding="utf-8")
        assert "<table" in html.lower()
        assert "</table>" in html.lower()
        assert "quality" in html.lower()

    def test_empty_folder_no_crash(self, tmp_path):
        from src.engine.quality_report import generate_quality_report
        report_path = generate_quality_report(tmp_path)
        assert isinstance(report_path, Path)
