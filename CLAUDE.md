# Synthetic Data Generator

Analytics-ready retail data generator (star-schema, inspired by ContosoRetailDW). Produces deterministic, idempotent datasets for BI, analytics, and data engineering.

## Tech Stack

- **Language:** Python 3.x
- **Web UI:** FastAPI backend (`web/api.py`) + React SPA (`web/frontend/`)
- **Data:** pandas, numpy, pyarrow, deltalake, fastparquet
- **Config:** PyYAML, pydantic
- **Other:** Faker (names), yfinance (FX rates), multiprocessing (parallel sales)

## Commands

```bash
# Run generation (CLI)
python main.py --format parquet --sales-rows 1000000
python main.py --format csv --customers 5000 --sales-rows 100000
python main.py --format deltaparquet --workers 8

# Useful flags
python main.py --dry-run                        # validate config, no generation
python main.py --only dimensions                # generate dims only
python main.py --only sales                     # generate sales only (dims must exist)
python main.py --regen-dimensions products      # force-rebuild specific dim
python main.py --regen-dimensions all           # force-rebuild all dims
python main.py --refresh-fx-master              # top up FX rates via Yahoo Finance

# Web UI
python -m uvicorn web.api:app --port 8502
```

## Architecture

Two-stage pipeline: **Dimensions** (sequential, dependency-aware) then **Facts** (parallel, multi-worker).

```
config.yaml + models.yaml --> Pipeline Runner
  |-- Dimensions Stage (sequential)
  |   generates ~20 dimension tables (customers, products, stores, dates, etc.)
  |-- Sales Stage (parallel via multiprocessing pool)
  |   |-- sales transactions (chunked, imap_unordered)
  |   |-- sales returns (optional)
  |   |-- budget forecasts (streamed accumulator)
  |   |-- inventory snapshots (streamed accumulator)
  |-- Package Output (final timestamped folder with dims, facts, SQL, Power BI project)
```

## Directory Map

```
main.py                          # Entry point, calls src.cli.main()
src/
  cli.py                         # Argument parser, CLI overrides
  dimensions/                    # One generator per dimension table
    customers.py, products/, stores.py, employees.py, dates.py,
    currency.py, exchange_rates.py, promotions.py, geography.py, etc.
  facts/
    sales/                       # Core sales fact generation
      sales.py                   # Entry point, orchestrates worker pool
      sales_logic/               # Business logic (orders, pricing, promos, allocation)
        globals.py               # State class (per-worker, sealed after bind)
        core/                    # customer_sampling, orders, promotions, allocation, pricing, delivery
      sales_models/              # quantity model (Poisson), pricing pipeline (inflation+markdown+snap)
      sales_worker/              # Multiprocessing pool, task definitions, schemas
      sales_writer/              # Output writers (parquet merge, delta, CSV), encoding
    budget/                      # Budget fact (accumulator + engine, Low/Medium/High scenarios)
    inventory/                   # Inventory snapshots (monthly grain, ABC classification, shrinkage)
  engine/
    config/                      # config_loader.py, config.py (normalizer registry, strict validation)
    packaging/                   # csv_packager, parquet_packager, delta_packager, SQL script gen
    runners/
      pipeline_runner.py         # Main orchestrator (override precedence, config injection)
      dimensions_runner.py       # Dependency-aware dim generation with version tracking
      sales_runner.py            # Sales pipeline coordinator
    powerbi_packaging.py         # Auto-generates .pbip project
  integrations/
    fx_yahoo.py                  # Yahoo Finance FX rate downloader
  tools/sql/                     # SQL Server script generators (CREATE TABLE, BULK INSERT)
  utils/
    shared_arrays.py             # Numpy shared memory for dimension broadcasting to workers
    static_schemas.py            # Column definitions for all output tables
    customer_profiles.py         # Acquisition profiles (steady/gradual/aggressive/instant)
    name_pools.py, output_utils.py, logging_utils.py, config_helpers.py
  versioning/
    version_checker.py           # Hash-based dimension version tracking (.version files)
web/
  api.py                         # FastAPI backend (SSE streaming, /generate, /config, /models)
  frontend/                      # React SPA (YAML editor, presets, log viewer)
```

## Dual Config System

### config.yaml -- Shape & Scale
Controls row counts, entity counts, date ranges, output format, parallelism, feature toggles.
- Key sections: `scale`, `defaults`, `sales`, `returns`, `products`, `customers`, `stores`, `employees`, `dates`, `exchange_rates`, `budget`, `inventory`
- Validated by strict normalizer registry in `src/engine/config/config.py`

### models.yaml -- Sales Behavior
Controls demand curves, pricing dynamics, quantity distribution, markdown rules, return reasons.
- Key sections: `models.macro_demand`, `models.quantity`, `models.pricing`, `models.brand_popularity`, `models.returns`
- Not overridable via CLI; edit directly or use web UI

### Override Precedence
CLI flags > config.yaml values (one-time, not persisted).

## Critical Gotchas

1. **Pricing injection:** At pipeline startup, `models.yaml` pricing appearance rules are injected into config's product section. Both product dimension generation and sales-time pricing use the same price grid. If you edit pricing bands in models.yaml, run `--regen-dimensions products` to sync.

2. **Dimension versioning:** Each dim has a `.version` file (JSON hash of its config section). Dims only regenerate if the hash changes or `--regen-dimensions` forces it. Stale outputs usually mean the version hash didn't change.

3. **State class is sealed:** `State` in `sales_logic/globals.py` is per-worker, initialized once via `bind_globals()`. Never mutate it after binding. Pass runtime tweaks through function args or config.

4. **Returns conditional skip:** Returns are silently skipped if `returns.enabled=true` AND `sales_output='sales'` AND `skip_order_cols=true` (returns need SalesOrderNumber to link back). A warning is logged.

5. **FX dates are coupled:** exchange_rates date range is overridden at runtime to match `defaults.dates.start/end`. You cannot set FX dates independently.

6. **Shared arrays:** Dimensions are loaded into numpy shared memory for zero-copy worker access. If adding new dimension columns, update `SharedArrayGroup` initialization.

7. **Scratch vs final output:** Workers write to scratch (`data/`), then packaging copies to final timestamped folder (`generated_datasets/`). Interrupted runs may leave scratch behind.

## Testing

```bash
# Run all tests (uses pyproject.toml defaults: -v --tb=short)
pytest

# Single file
pytest tests/test_geography.py

# Single class
pytest tests/test_config_loader.py::TestParseDate

# Single test
pytest tests/test_config_loader.py::TestParseDate::test_iso_string

# Keyword filter (supports and/or/not)
pytest -k "deterministic"
pytest -k "geography and not key"

# Useful flags
pytest -x              # stop on first failure
pytest -s              # show print() output
pytest --tb=long       # full tracebacks
pytest --lf            # rerun only last-failed tests
pytest --co            # list tests without running
```

Test files: `tests/test_config_loader.py`, `test_pricing_pipeline.py`, `test_quantity_model.py`, `test_geography.py`, `test_customer_profiles.py`, `test_version_store.py`, `test_state.py`, `test_determinism.py` (186 tests total).

## Output Formats

- **CSV:** chunked files + auto-generated SQL Server scripts (DDL, BULK INSERT, constraints, views)
- **Parquet:** merged single file (configurable) + compression/row_group tuning
- **Delta Lake (deltaparquet):** partitioned by Year/Month, ACID transactions via delta-rs
- **All formats:** auto-generate Power BI `.pbip` project with pre-configured relationships

## Workflow Tips

- Use `--dry-run` before large runs to validate config
- Use `--only dimensions` then `--only sales` to iterate on each stage independently
- `workers` defaults to CPU count; don't exceed it
- `chunk_size` too small = overhead, too large = memory pressure (default 1M is good)
- Check `generated_datasets/` for timestamped output folders
