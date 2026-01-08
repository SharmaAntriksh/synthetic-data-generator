# FakeDataGenerator (Contoso-based)

Generate a complete, **analytics-ready retail dataset** inspired by **ContosoRetailDW**, with scalable products and realistic sales behavior suitable for BI, analytics, and modeling scenarios.

The generator is designed to be **deterministic**, **schema-stable**, and **idempotent**, making it appropriate for repeatable demos, training, and benchmarking.

---

## What this generator produces

### Dimensions

* Geography
* Customers
* Stores
* Dates
* Currency
* Exchange Rates
* ProductCategory *(static reference)*
* ProductSubcategory *(static reference)*
* Product *(scalable, priced)*

### Facts

* Sales (orders, order lines, discounts, promotions)

---

## Prerequisites

* **Python 3.10 or later**
* Git

Optional:

* Power BI Desktop (for analysis and modeling)

Verify your Python version:

```bash
python --version
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SharmaAntriksh/contoso-fake-data-generator.git
cd contoso-fake-data-generator
```

---

## Quick start (Windows – recommended)

If you are on **Windows**, the fastest way to get started is using the provided PowerShell scripts.

### One-time setup

This creates a virtual environment and installs all dependencies:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

> The execution policy change applies **only to the current PowerShell session**.

You only need to run this once.

### Configure your run

Open **`run.ps1`** and edit the parameters at the top:

```powershell
$Format        = "deltaparquet"
$SalesRows     = 102500
$Customers     = 10000
$Products      = 6500
$StartDate     = "2024-01-01"
$EndDate       = "2024-12-31"
```

### Run the generator

```powershell
.\run.ps1
```

This is functionally equivalent to running `python main.py` with a full set of CLI arguments.

---

## Manual setup (cross-platform)

Use this approach on **macOS**, **Linux**, or if you prefer a manual setup.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Edit **`config.yaml`** to control data volume, pricing, and regeneration behavior.

### Minimal working example

```yaml
products:
  use_contoso_products: false
  num_products: 50000
  seed: 42

  pricing:
    base:
      value_scale: 1.0
      min_unit_price: 10
      max_unit_price: 5000

sales:
  force_regenerate: true
```

### Key options

* **use_contoso_products**

  * `true`  → use original Contoso products only
  * `false` → expand products up to `num_products`

* **num_products**
  Total number of products when expansion is enabled

* **value_scale**
  Global price multiplier (inflation / deflation scenarios)

---

## Run the generator

Below is an example of a **full production-style run** using explicit CLI arguments.
This is equivalent to the parameters defined in `run.ps1`.

```bash
python main.py \
  --format parquet \
  --row-group-size 500000 \
  --skip-order-cols \
  --sales-rows 102500 \
  --customers 10000 \
  --stores 300 \
  --products 6500 \
  --promotions 150 \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --workers 6 \
  --chunk-size 2000000 \
  --clean
```

The pipeline is **idempotent**:

* Dimensions are skipped if already generated
* Regeneration occurs only when `force_regenerate: true` is set

---

## Output

Generated data is written to the `generated_datasets/` directory:

```
2026-01-08 11_47_50 PM Customers 10K Sales 102K Parquet/
├── dimensions/
│   ├── product_category.parquet
│   ├── product_subcategory.parquet
│   ├── product.parquet
│   └── ...
└── facts/
    └── sales.parquet
```

* Default format: **Parquet**
* CSV and SQL helpers are available under `tools/sql/`
