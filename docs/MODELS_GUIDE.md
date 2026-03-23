# Models Reference (`models.yaml`)

This file controls product-level sales behavior: basket sizes, pricing dynamics, brand popularity, and return patterns. Unlike `config.yaml` (which controls shape and scale), `models.yaml` controls *how* sales behave at the transaction level.

Customer behavior (lifecycle, demand shape, growth curves) is controlled by `customers.profile` in `config.yaml`, not here.

> **Note:** `models.yaml` is not overridable via CLI flags. Edit it directly or use the web UI's Models tab.

---

## Macro Demand

Controls the overall demand curve across the simulation period. These factors multiply on top of the customer acquisition profile to shape total sales volume over time.

| Key | Description | Example |
|-----|-------------|---------|
| `macro_demand.yearly_growth` | Smooth compound annual growth rate applied to sales volume. 0.08 = 8% year-over-year growth in total demand. | `0.08` |
| `macro_demand.row_share_of_growth` | How much of the growth is expressed as more rows (transactions) vs. higher quantities per order. 1.0 = all growth comes from more transactions. 0.5 = half from more transactions, half from larger baskets. | `1.0` |
| `macro_demand.shock_probability` | Per-month probability of a random demand shock (sudden spike or dip). 0.0 = smooth demand. 0.03 = ~3% chance each month of an unexpected swing. | `0.0` |
| `macro_demand.seasonality_amplitude` | Magnitude of within-year seasonal swings. 0.08 = ~8% peak-to-trough variation driven by month-of-year patterns (e.g., holiday uplift in Nov/Dec). | `0.08` |
| `macro_demand.noise_std` | Standard deviation of random month-to-month noise layered on top of the seasonal curve. Keeps the demand curve from looking too smooth. | `0.02` |

### Year-level factors

Per-year multipliers applied on top of the baseline demand curve. Use these to model specific business events like a recession year, a product launch year, or a market expansion.

| Key | Description | Example |
|-----|-------------|---------|
| `macro_demand.year_level_factors.mode` | How the factor list is applied. `"once"` = each value maps to one year in sequence. `"cycle"` = the list repeats if the simulation has more years than values. | `"once"` |
| `macro_demand.year_level_factors.values` | List of multipliers, one per year. 1.0 = follow the baseline. <1.0 = demand dip (e.g., 0.85 = 15% below baseline). >1.0 = demand peak (e.g., 1.2 = 20% above baseline). | `[1.0, 1.0, 1.0, 1.0, 1.0]` |

---

## Quantity

Controls basket size — how many items a customer buys per order line.

| Key | Description | Example |
|-----|-------------|---------|
| `quantity.base_poisson_lambda` | Mean of the Poisson distribution used to sample raw quantity. Higher values = larger typical baskets. 2.1 means most orders have 1-3 items per line. | `2.1` |
| `quantity.min_qty` | Floor clamp on quantity. No order line will have fewer than this many items. | `1` |
| `quantity.max_qty` | Ceiling clamp on quantity. Prevents unrealistically large single-line orders. | `4` |
| `quantity.noise_sigma` | Standard deviation of multiplicative noise applied to the monthly-adjusted lambda. Adds randomness so not every order in the same month has the same basket size distribution. | `0.03` |

### Monthly factors

Twelve multipliers (Jan-Dec) applied to `base_poisson_lambda` to create seasonal basket size variation. Values > 1.0 increase basket sizes that month; < 1.0 decrease them.

```yaml
quantity.monthly_factors:
  - 0.99    # Jan — post-holiday lull
  - 0.98    # Feb — lowest basket sizes
  - 1.00    # Mar
  - 1.00    # Apr
  - 1.01    # May
  - 1.02    # Jun — summer buying
  - 1.02    # Jul
  - 1.01    # Aug — back to school
  - 1.00    # Sep
  - 1.03    # Oct — pre-holiday build
  - 1.06    # Nov — Black Friday / holiday peak
  - 1.05    # Dec — holiday season
```

---

## Pricing

Controls how prices evolve over time and how they appear in the output. Three sub-systems work together: inflation drift, markdown discounts, and appearance (rounding/snapping).

### Inflation

Simulates gradual price increases over time. Each product's `UnitPrice` drifts upward from its base `ListPrice` according to these parameters.

| Key | Description | Example |
|-----|-------------|---------|
| `pricing.inflation.annual_rate` | Annual inflation rate applied to unit prices. 0.10 = 10% per year. Compounded monthly — a product priced at $100 in year 1 costs ~$110 in year 2. | `0.10` |
| `pricing.inflation.month_volatility_sigma` | Monthly noise on the inflation factor. Adds micro-variation so prices don't increase in a perfectly smooth line. | `0.003` |
| `pricing.inflation.factor_clip` | `[min, max]` bounds on the cumulative inflation factor. Prevents prices from deflating below the floor or inflating beyond the ceiling. `[1.00, 1.50]` = prices can rise up to 50% but never drop below the original. | `[1.00, 1.50]` |
| `pricing.inflation.volatility_seed` | Random seed for inflation noise. Change this to get a different inflation trajectory with the same parameters. | `123` |

### Markdown

Controls discount amounts applied to individual sales transactions. Not all sales get a discount — the ladder defines the probability distribution.

| Key | Description | Example |
|-----|-------------|---------|
| `pricing.markdown.enabled` | Master toggle for markdowns. When `false`, all sales are at full price (DiscountAmount = 0). | `true` |
| `pricing.markdown.max_pct_of_price` | Maximum discount as a fraction of unit price. 0.40 = discounts can't exceed 40% off. | `0.40` |
| `pricing.markdown.min_net_price` | Floor on the net price after discount. Prevents items from being sold for less than this amount. | `0.01` |
| `pricing.markdown.allow_negative_margin` | When `false`, discounts are clamped so that `NetPrice` never drops below `UnitCost`. Prevents selling at a loss. | `false` |

#### Discount ladder

A weighted list of discount tiers. Each sale randomly picks one tier based on the weights. Weights must sum to 1.0.

```yaml
pricing.markdown.ladder:
  - { kind: none, value: 0.0,   weight: 0.55 }   # 55% of sales: no discount
  - { kind: pct,  value: 0.05,  weight: 0.14 }   # 14% of sales: 5% off
  - { kind: pct,  value: 0.10,  weight: 0.13 }   # 13% of sales: 10% off
  - { kind: pct,  value: 0.15,  weight: 0.08 }   #  8% of sales: 15% off
  - { kind: pct,  value: 0.20,  weight: 0.05 }   #  5% of sales: 20% off
  - { kind: pct,  value: 0.25,  weight: 0.03 }   #  3% of sales: 25% off
  - { kind: pct,  value: 0.30,  weight: 0.02 }   #  2% of sales: 30% off
```

- `kind: none` — no discount applied
- `kind: pct` — percentage discount (value is the fraction, e.g., 0.10 = 10% off)

### Appearance

Controls how prices are rounded and snapped to look realistic. Without this, prices would have arbitrary decimal values (e.g., $47.3291). With appearance rules, they snap to values like $47.99 or $45.00.

| Key | Description | Example |
|-----|-------------|---------|
| `pricing.appearance.enabled` | Master toggle. When `false`, prices keep their raw calculated values without any rounding or snapping. | `true` |

Each price column (`unit_price`, `unit_cost`, `discount`) has its own appearance rules:

#### Rounding

| Key | Description | Options |
|-----|-------------|---------|
| `rounding` | How to round the price before applying the ending. | `floor` (round down), `nearest` (round to nearest), `ceil` (round up) |

#### Endings

Defines the decimal portion of the final price. A weighted list — if multiple endings are defined, one is chosen randomly per transaction.

```yaml
endings:
  - { value: 0.99, weight: 1.0 }   # all prices end in .99 (e.g., $49.99)
```

Common endings: `0.99` (psychological pricing), `0.00` (clean prices), `0.95`, `0.49`.

#### Price bands

Controls the step size for rounding based on the price magnitude. Cheaper items get finer rounding; expensive items get coarser rounding.

```yaml
bands:
  - { max: 100,  step: 1 }    # prices under $100: round to nearest $1
  - { max: 500,  step: 5 }    # $100-$500: round to nearest $5
  - { max: 2000, step: 10 }   # $500-$2000: round to nearest $10
  - { max: 5000, step: 25 }   # $2000-$5000: round to nearest $25
  - { max: 1e18, step: 50 }   # above $5000: round to nearest $50
```

> **Important:** If you change appearance rules, run `--regen-dimensions products` to sync product dimension prices. Both the product dimension and sales-time pricing use the same price grid.

---

## Brand Popularity

Simulates a "winner" brand that rotates each year. The winning brand gets a demand boost, creating realistic brand-level trends in the sales data.

| Key | Description | Example |
|-----|-------------|---------|
| `brand_popularity.enabled` | Master toggle. When `false`, all brands have equal demand weight. | `true` |
| `brand_popularity.seed` | Random seed for selecting the winning brand each year. | `123` |
| `brand_popularity.winner_boost` | Demand multiplier for the winning brand. 1.4 = the hot brand gets 40% more sales than its baseline. Other brands are slightly suppressed to keep totals stable. | `1.4` |

---

## Returns

Controls the return reason distribution and return timing. The overall return rate is set in `config.yaml` (`returns.return_rate`); this section controls the *characteristics* of those returns.

| Key | Description | Example |
|-----|-------------|---------|
| `returns.enabled` | Master toggle for the returns model. When `false`, returns use default reason distribution. | `true` |

### Reasons

A weighted list of return reasons. Each return event picks one reason based on the weights. Weights must sum to 1.0.

```yaml
returns.reasons:
  - { key: 1, label: "Damaged / Defective",    weight: 0.28 }   # most common
  - { key: 2, label: "Wrong Item",              weight: 0.12 }
  - { key: 3, label: "Not as Described",        weight: 0.14 }
  - { key: 4, label: "Arrived Late",            weight: 0.08 }
  - { key: 5, label: "Better Price Elsewhere",  weight: 0.06 }
  - { key: 6, label: "No Longer Needed",        weight: 0.20 }   # second most common
  - { key: 7, label: "Size / Fit",              weight: 0.07 }
  - { key: 8, label: "Other",                   weight: 0.05 }
```

- `key` — integer key written to the `ReturnReasonKey` column (joins to the `ReturnReason` dimension)
- `label` — human-readable reason text written to the dimension table
- `weight` — probability of this reason being selected

### Return timing

Controls the delay between the sale date and the return date.

| Key | Description | Example |
|-----|-------------|---------|
| `returns.lag_days.distribution` | Statistical distribution for the return delay. `"triangular"` produces a realistic right-skewed shape (most returns happen quickly, a long tail of late returns). | `"triangular"` |
| `returns.lag_days.mode` | Mode (peak) of the triangular distribution in days. 7 = most returns happen about a week after purchase. The min/max bounds come from `config.yaml` (`returns.min_days_after_sale` / `returns.max_days_after_sale`). | `7` |

### Return quantity

| Key | Description | Example |
|-----|-------------|---------|
| `returns.quantity.full_line_probability` | Probability that a return covers the full quantity of the original order line. 0.85 = 85% of returns are full-line returns; 15% are partial returns (random fraction of the original quantity). | `0.85` |
