# FakeDataGenerator (Contoso-based)

Generate a complete, analytics-ready retail dataset based on **ContosoRetailDW**, with scalable products and realistic sales behavior.

---

## What this generator produces

**Dimensions**
- Geography  
- Customers  
- Stores  
- Dates  
- Currency  
- Exchange Rates  
- ProductCategory (static reference)  
- ProductSubcategory (static reference)  
- Product (scalable, priced)

**Facts**
- Sales (orders, order lines, discounts, promotions)

---

## Prerequisites

- Python **3.10+**
- Git

(Optional) Power BI Desktop for analysis.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SharmaAntriksh/contoso-fake-data-generator.git
cd FakeDataGenerator
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Edit `config.yaml`.

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

- **use_contoso_products**
  - `true`  → use original Contoso products only
  - `false` → expand products up to `num_products`

- **num_products**
  - Total number of products when expansion is enabled

- **value_scale**
  - Global price multiplier (inflation / deflation)

---

## Run the generator

From the project root:

```bash
python main.py
```

The pipeline is **idempotent**.  
Dimensions are skipped if already generated unless `force_regenerate` is enabled.

---

## Output

Generated data is written to the `output/` folder:

```
output/
├── dimensions/
│   ├── product_category.parquet
│   ├── product_subcategory.parquet
│   ├── product.parquet
│   └── ...
└── facts/
    └── sales.parquet
```

- Default format: **Parquet**
- CSV and SQL helpers are available under `tools/sql/`

---

## Guarantees

- ProductCategory and ProductSubcategory are static reference data
- Products always include:
  - `ProductKey`
  - `BaseProductKey`
  - `VariantIndex`
- Product pricing is defined at product generation time
- Sales only applies discounts and promotions
- Schemas are stable across runs

---

## Common issues

**Prices look lower than expected**  
Check `pricing.base.value_scale` in `config.yaml`.

**Data is not regenerating**  
Set `force_regenerate: true` in the relevant config section.

---

## Next steps

- Load Parquet files into Power BI
- Use `BaseProductKey` to analyze product variants
- Generate SQL DDL using `tools/sql/generate_create_table_scripts.py`
