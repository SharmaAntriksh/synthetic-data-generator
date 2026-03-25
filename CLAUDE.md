# Synthetic Data Generator

Analytics-ready retail data generator (star-schema, inspired by ContosoRetailDW). Produces deterministic, idempotent datasets for BI, analytics, and data engineering.

## Tech Stack

- **Language:** Python 3.x
- **Web UI:** FastAPI backend (`web/api.py` + `web/routes/`) + React SPA (`web/frontend/`)
- **Data:** pandas, numpy, pyarrow, deltalake
- **Config:** PyYAML, pydantic
- **Other:** yfinance (FX rates), multiprocessing (parallel sales)

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
  |   generates ~22 dimension tables (customers, products, stores, dates, plans, etc.)
  |-- Sales Stage (parallel via multiprocessing pool)
  |   |-- sales transactions (chunked, imap_unordered)
  |   |-- sales returns (optional)
  |   |-- budget forecasts (streamed accumulator)
  |   |-- inventory snapshots (streamed accumulator)
  |   |-- customer wishlists (streamed accumulator)
  |   |-- customer complaints (streamed accumulator)
  |-- Package Output (final timestamped folder with dims, facts, SQL, Power BI project)
```

## Directory Map

```
main.py                          # Entry point, calls src.cli.main()
src/
  cli.py                         # Argument parser, CLI overrides
  defaults.py                    # Centralized hardcoded constants (stores, customers, promos, etc.)
  exceptions.py                  # Custom exception hierarchy (PipelineError, ConfigError, etc.)
  dimensions/                    # One generator per dimension table
    customers/                   # Customer dimension package
      __init__.py                # Re-exports run_customers, run_subscriptions
      generator.py               # Main orchestrator: generate_synthetic_customers(), run_customers()
      helpers.py                 # Date math, acquisition curves, income/phone/credit/address generation
      org_profile.py             # OrganizationProfile constants + generate_org_profile()
      households.py              # Household assignment (spouse/dependent matching)
      scd2.py                    # SCD2 life event engine (career, marriage, relocation, etc.)
      subscriptions.py           # Plans + CustomerSubscriptions (many-to-many bridge)
      worker.py                  # Multiprocessing chunk worker for parallel customer generation
    dates/                       # Date dimension package — one module per calendar system
      __init__.py                # Re-exports run_dates, generate_date_table, resolve_date_columns, WeeklyFiscalConfig, run_time_table
      runner.py                  # run_dates() — config parsing, versioning, parquet write
      generator.py               # Orchestrator: calls add_*_columns() subsystems in sequence
      calendar.py                # add_calendar_columns() — base calendar + as-of relative offsets
      iso.py                     # add_iso_columns() — ISO week fields + offsets
      fiscal.py                  # add_fiscal_columns() — monthly fiscal system + offsets
      weekly_fiscal.py           # WeeklyFiscalConfig + add_weekly_fiscal_columns() — 4-4-5 system
      columns.py                 # resolve_date_columns() — builds ordered output column list per config
      helpers.py                 # Shared constants (_EXCEL_EPOCH, _ISO_WEEK_REF) and utility functions
      time.py                    # Time-of-day dimension (15-minute grain, 96 rows)
    employees/                   # Employee dimension package
      __init__.py                # Re-exports run_employees, run_employee_store_assignments
      generator.py               # Employee generation with role profiles, HR fields
      employee_store_assignments.py  # Multi-store assignment with transfer reasons
      transfers.py               # Transfer scheduling logic
    exchange_rates/              # Currency + exchange rate dimensions
      __init__.py                # Re-exports run_currency, run_exchange_rates
      currency.py                # Currency dimension from defaults
      exchange_rates.py          # Daily exchange rates (Yahoo Finance or synthetic)
      monthly_rates.py           # Monthly aggregated rates
      helpers.py                 # Triangulation, date alignment utilities
    products/                    # Product dimension package
      __init__.py                # Re-exports run_suppliers, generate_product_dimension, load_static_dimension
      products.py                # Entry point: generate_product_dimension() (category + subcategory + products)
      generator.py               # Orchestrator: load → expand → price → enrich → SCD2 → write
      scd2.py                    # SCD2 price revision versioning (vectorized)
      product_profile.py         # ProductProfile enrichment (analytical attributes)
      pricing.py                 # Price grid: bands, margin, snap, brand normalization
      contoso_loader.py          # Load base Contoso product catalog
      contoso_expander.py        # Stratified trim/expand of Contoso products
      catalog_builder.py         # Product catalog assembly helpers
      static_loader.py           # Generic static dimension loader (category, subcategory)
      suppliers.py               # Supplier dimension generation
      worker.py                  # Multiprocessing chunk worker for parallel product enrichment
    stores/                      # Store dimension package
      __init__.py                # Re-exports run_stores
      generator.py               # Store generation with geography, staffing, assortment
    geography.py                 # Geography dimension (countries, states, cities, population weighting)
    promotions.py                # Promotions dimension (seasonal, clearance, BOGO, etc.)
    lookups.py                   # Lookup dimensions (sales channels, loyalty tiers, acquisition channels)
    return_reasons.py            # Return reason dimension
  facts/
    sales/                       # Core sales fact generation
      sales.py                   # Entry point, orchestrates worker pool
      sales_logic/               # Business logic (orders, pricing, promos, allocation)
        globals.py               # State class (per-worker, sealed after bind) + SalesContext dataclass
        core/                    # customer_sampling, orders, promotions, allocation, pricing, delivery
      sales_models/              # quantity model (Poisson), pricing pipeline (inflation+markdown+snap)
      sales_worker/              # Multiprocessing pool, task definitions, schemas
      sales_writer/              # Output writers (parquet merge, delta, CSV), encoding
    budget/                      # Budget fact (accumulator + engine, Low/Medium/High scenarios)
    inventory/                   # Inventory snapshots (monthly grain, ABC classification, shrinkage)
    wishlists/                   # Customer wishlists (streamed from sales worker accumulators)
    complaints/                  # Customer complaints (streamed from sales worker accumulators)
  engine/
    config/                      # config_loader.py, config.py (normalizer registry), config_schema.py (Pydantic models)
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
    config_helpers.py            # Canonical type coercion helpers (bool_or, int_or, float_or, etc.)
    config_precedence.py         # Standardized config resolution (resolve_seed, resolve_dates)
    name_pools.py, output_utils.py, logging_utils.py
  versioning/
    version_checker.py           # Hash-based dimension version tracking (.version files)
web/
  api.py                         # FastAPI app entry point (includes routers, serves frontend)
  shared_state.py                # Shared mutable state and helpers for web layer
  routes/                        # FastAPI routers (split from monolithic api.py)
    config_routes.py             # /api/config endpoints
    models_routes.py             # /api/models endpoints
    generation_routes.py         # /api/generate, /api/validate endpoints
    presets_routes.py            # /api/presets endpoints
    data_routes.py               # /api/datasets endpoints (browse, preview generated data)
    import_routes.py             # /api/import endpoints (SQL Server import management)
  frontend/                      # React SPA (YAML editor, presets, log viewer)
```

## Dual Config System

### config.yaml -- Shape & Scale
Controls row counts, entity counts, date ranges, output format, parallelism, feature toggles.
- Key sections: `scale`, `defaults`, `sales`, `returns`, `products`, `customers`, `stores`, `employees`, `dates`, `exchange_rates`, `budget`, `inventory`, `subscriptions`, `wishlists`, `complaints`
- Validated by strict normalizer registry in `src/engine/config/config.py`

### models.yaml -- Sales Behavior
Controls demand curves, pricing dynamics, quantity distribution, markdown rules, return reasons.
- Key sections: `models.macro_demand`, `models.quantity`, `models.pricing`, `models.brand_popularity`, `models.returns`
- Validated through Pydantic at pipeline startup: `ModelsConfig.from_raw_dict()` → `ModelsInnerConfig` (typed sub-models for each section)
- `State.models_cfg` holds the validated `ModelsInnerConfig` instance (not a raw dict)
- `resolve_customer_profile()` may replace `macro_demand` and `customers` sub-models with plain dicts at runtime — downstream code must handle both via `.get()` shims
- Not overridable via CLI; edit directly or use web UI

### Override Precedence
CLI flags > config.yaml values (one-time, not persisted).

## Critical Gotchas

1. **Pricing injection:** At pipeline startup, `models.yaml` pricing appearance rules are injected into config's product section. Both product dimension generation and sales-time pricing use the same price grid. If you edit pricing bands in models.yaml, run `--regen-dimensions products` to sync.

2. **Dimension versioning:** Each dim has a `.version` file (JSON hash of its config section). Dims only regenerate if the hash changes or `--regen-dimensions` forces it. Stale outputs usually mean the version hash didn't change.

3. **State class is sealed via metaclass:** `State` in `sales_logic/globals.py` uses `_SealableMeta` metaclass to enforce immutability after `bind_globals()` calls `seal()`. Any `setattr(State, ...)` after sealing raises `RuntimeError`. Never mutate it after binding. For new code, prefer `SalesContext.from_state()` (immutable dataclass snapshot) over accessing `State` directly. In tests, call `State.reset()` to unseal.

4. **Returns conditional skip:** Returns are auto-disabled at config load time if `returns.enabled=true` AND `sales_output='sales'` AND `skip_order_cols=true` (returns need SalesOrderNumber to link back). A warning is logged by `apply_cross_section_rules()` in `config.py`.

5. **FX dates are coupled:** exchange_rates date range is overridden at runtime to match `defaults.dates.start/end`. You cannot set FX dates independently.

6. **Shared arrays:** Dimensions are loaded into numpy shared memory for zero-copy worker access. If adding new dimension columns, update `SharedArrayGroup` initialization.

7. **Scratch vs final output:** Workers write to scratch (`data/`), then packaging copies to final timestamped folder (`generated_datasets/`). Interrupted runs may leave scratch behind.

8. **Exception hierarchy & specificity:** Use exceptions from `src/exceptions.py` (`ConfigError`, `DimensionError`, `SalesError`, `PackagingError`, `ValidationError`) instead of generic `RuntimeError`/`ValueError`. All inherit from `PipelineError`. Avoid bare `except Exception`— always catch specific types (e.g., `except (KeyError, ValueError, OSError)`).

9. **Type coercion helpers:** Import `bool_or`, `int_or`, `float_or`, `str_or` from `src.utils.config_helpers`. Do not define local variants. These are still required for plain dict accesses (worker configs, nested profile dicts) and function parameters that might be None. For Pydantic model attributes with typed defaults, they're redundant but harmless — new code accessing Pydantic fields can skip them.

10. **Hardcoded constants:** Domain constants (store types, customer demographics, promo categories, currency maps, etc.) live in `src/defaults.py`. Dimension generators import from there — do not inline large constant dicts. Probability arrays are validated at import time to sum to 1.0.

11. **API versioning:** All web API endpoints are available at both `/api/...` (backward-compatible) and `/v1/api/...` (versioned). Both routes hit the same router handlers.

12. **Web layer thread safety:** Shared mutable state in `web/shared_state.py` (`_cfg`, `_models_cfg`, etc.) is guarded by `_cfg_lock`. Always acquire the lock when reading or mutating these globals from route handlers. Reads must `copy.deepcopy()` under the lock to avoid races.

13. **int32 overflow in ID arithmetic:** Order ID math (add, multiply) must use `int64` intermediates before casting back to `int32`. Silent wraparound at 2^31 produces negative/duplicate IDs.

14. **numpy bincount weights dtype:** `np.bincount(..., weights=...)` requires `float64` weights. Passing `int8` silently overflows at >127 elements, producing wrong counts.

15. **CDF + searchsorted boundary:** After computing a CDF via `np.cumsum(w) / total`, always clamp `cdf[-1] = 1.0`. Floating-point rounding can leave the last element slightly below 1.0, causing `searchsorted` to return out-of-bounds indices.

16. **SQL output escaping:** Use `N'...'` (Unicode prefix) for file paths in `BULK INSERT FROM` statements. Escape single quotes in names passed to `OBJECT_ID()`. Without this, non-ASCII paths or names with apostrophes break generated SQL.

17. **TMDL expression injection:** File paths embedded in Power BI M expressions must have `"` escaped to `\"`. Unescaped quotes in paths break the generated `.tmdl` files.

18. **CORS allowlist:** `web/api.py` restricts origins to `localhost:8502` and `localhost:3000` (not `*`). A `SecurityHeadersMiddleware` adds `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, and `Referrer-Policy` to all responses.

19. **Web payload size limits:** YAML and models endpoints reject bodies > 1 MB (HTTP 413). This prevents memory exhaustion from oversized payloads.

20. **Weekly fiscal column guard:** `dates/weekly_fiscal.py` only computes `FWYearWeekOffset` / `FWYearMonthOffset` / `FWYearQuarterOffset` when `FWYearWeekNumber` exists in the DataFrame. Previously, accessing these columns when `include_weekly_fiscal=false` caused a `KeyError`.

21. **Nested probability validation:** `defaults.py` now validates probability arrays inside nested dicts (`CUSTOMER_HOME_OWNERSHIP_PROBS_BY_INCOME`, `CUSTOMER_OCCUPATION_PROBS_BY_EDUCATION`) and lists (`CUSTOMER_MARITAL_PROBS_BY_AGE`, `CUSTOMER_EDUCATION_PROBS_BY_AGE`) at import time, not just top-level arrays.

22. **Log file descriptor thread safety:** `_ensure_log_file_open()` in `logging_utils.py` is guarded by `_LOG_FD_LOCK` to prevent races when multiple threads open the log file simultaneously.

23. **Both configs are Pydantic, not dicts:** `cfg` is `AppConfig` and `models_cfg` (aka `State.models_cfg`) is `ModelsInnerConfig` — both Pydantic models from `src/engine/config/config_schema.py`. Access fields via attribute syntax, not dict syntax. Key rules:
    - **Read config:** `cfg.sales.file_format` or `models_cfg.pricing.inflation` — attribute access, not `cfg["field"]`
    - **Write config:** `cfg.sales.file_format = "csv"` (attribute assignment) — bracket write (`cfg["key"] = val`) works via `_MutationMixin.__setitem__`
    - **`.get()` shim:** `_MutationMixin.get(key, default)` delegates to `getattr()`, so existing `.get()` chains work on Pydantic models. This is intentional for backward compatibility with code that handles both dicts and models
    - **`resolve_customer_profile` caveat:** After profile resolution, `models_cfg.macro_demand` and `models_cfg.customers` may be plain dicts (replaced via `__setitem__`). Code accessing these must handle both Pydantic models and dicts — `.get()` works on both
    - **`_`-prefixed keys are stripped:** Pydantic v2 treats `_`-prefixed keys as private fields. `_strip_internal_keys()` in `config_schema.py` removes normalizer metadata before validation
    - **`extra="forbid"` on sub-models:** All config sub-models (via `_Base`) reject unknown keys. `AppConfig` and `ModelsConfig` use `extra="allow"` for runtime-injected keys. Add new fields to the schema class if needed
    - **Worker configs are plain dicts:** `worker_cfg`, `acfg`, and sales worker internals remain plain dicts for multiprocessing pickling — these correctly use dict access
    - **Copying configs:** Use `cfg.model_copy(deep=True)` instead of `dict(cfg)` or `copy.deepcopy(cfg)`
    - **Hash functions:** Use `_to_dict(obj)` helper (calls `model_dump()` on Pydantic, pass-through for dicts) before `json.dumps()` for config hashing
    - **Schema location:** All Pydantic models live in `src/engine/config/config_schema.py`. `AppConfig.from_raw_dict(d)` and `ModelsConfig.from_raw_dict(d)` are the canonical constructors

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

Test files: `tests/test_config_loader.py`, `test_pricing_pipeline.py`, `test_quantity_model.py`, `test_geography.py`, `test_customer_profiles.py`, `test_version_store.py`, `test_state.py`, `test_determinism.py`, `test_integration.py`, `test_web_api.py`, `test_dimensions.py`, `test_packaging.py`, `test_sales_logic.py`, `test_utils.py`, `test_web_routes.py`, `test_schema.py`, `test_gotchas_and_guards.py` (914+ tests; web API/route tests require `httpx` and are skipped without it).

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
