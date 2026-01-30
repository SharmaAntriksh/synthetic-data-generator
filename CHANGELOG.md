# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

---
## v0.5.0 – Realistic data generation and improved SQL/Power BI compatibility

### Added
- Support for inactive dimension members during sales generation.
  - Customers and products can now be generated without corresponding sales.
  - Enables more realistic analytical scenarios.
- Optional SQL Server helper scripts included with CSV output:
  - Bootstrap scripts for table-valued types and stored procedures.
  - Optional PK/FK constraint scripts.
  - Optional clustered columnstore management script.
  - Helper scripts are copied only and are not executed by the generator.

### Changed
- Standardized schemas and data types across generated datasets and Power BI models.
- Boolean-style fields (e.g. IsActiveInSales) are now emitted as numeric flags (0/1).

### Fixed
- Improved Power BI / Power Query compatibility by aligning SQL data types.
- Prevented BULK INSERT and Power Query type inference issues during CSV → SQL imports.

---

## v0.4.0 – Power BI project packaging

### Added
- Include Power BI Project (PBIP) templates directly in final output folders
- Automatically update dataset folder paths inside PBIP using a shared `ContosoFolder` expression
- Support format-specific PBIP templates for CSV and Parquet outputs
- Enable opening Power BI projects without manual path configuration

### Fixed
- Skip PBIP packaging for `deltaparquet` output format

---

## v0.3.1 – Schema fixes and SQL import stability

### 🐛 Fixes
- Fixed SQL view creation failures on repeated imports by making all views idempotent
- Corrected `vw_Sales` generation to safely handle optional `SalesOrderNumber` and `SalesOrderLineNumber`
- Resolved SQL import errors caused by mismatches between generated schemas and SQL views

### 🧱 Schema & Data Model
- Normalized currency naming across Currency, Geography, and Sales logic
- Clarified Geography ↔ Currency joins to avoid semantic overloading
- Ensured all date-related columns in the Dates dimension are written as DATE (no DATETIME leakage)
- Aligned Contoso product category and subcategory schemas across Parquet and SQL outputs

### 🔧 Pipeline & Tooling
- Improved robustness of the CSV → SQL Server import workflow
- Reduced dependency on manual database cleanup between runs
- Better consistency between generated CSV, Parquet, and SQL artifacts

---

## [v0.3.0] – 2026-01-28

### ✨ Added
- End-to-end CSV to SQL Server workflow with automatic view creation
- Automatic inclusion of `create_views.sql` in CSV output for SQL consumers
- Adaptive SQL views that support optional order number columns

### 🛠 Improvements
- Cleaner output packaging for CSV runs
- More robust SQL Server import sequencing (views created after schema and data load)

---

## [v0.2.1] – 2026-01-11

### 🐛 Fixed
- Added missing `pyodbc` dependency required for SQL Server import tooling

---

## [v0.2.0] – 2026-01-11

### ✨ Added
- CSV to SQL Server import tooling with one-command execution
- Support for both Windows Authentication and SQL Authentication
- Safe import behavior that skips execution when the target database already exists

### 🎛 UI Improvements
- Clearer regeneration intent in the Generate UI
- Improved handling and visibility of forced dimension regeneration

### 📄 Documentation
- Overhauled README with structured setup, generation, and SQL Server import instructions
- Added copy-paste–ready examples for both authentication modes
- Improved guidance around CSV output and SQL import workflow

### 🛠 Improvements
- More predictable and repeatable data generation workflows
- Better separation between generator logic and optional import tooling

---

## [v0.1.0] – 2026-01-10

### 🎉 Initial Release
- Synthetic data generator based on Contoso-style retail schema
- Configurable dimensions and sales fact generation
- Support for CSV, Parquet, and Delta Parquet outputs
- Power BI templates and sample reports
- Web UI for interactive data generation
