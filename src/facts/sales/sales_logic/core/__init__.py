"""Sales core generation logic.

Pure(ish) numpy routines shared by the sales chunk builder.

This package was split from a single core.py module. All public symbols
are re-exported here so existing imports continue to work unchanged:

    from .core import compute_dates, build_orders, ...
"""

from __future__ import annotations

# Re-export: date logic
from .delivery import (
    fmt,
    _yyyymmdd_from_days,
    _stable_row_hash,
    compute_dates,
)

# Re-export: order logic
from .orders import (
    build_month_demand,
    build_orders,
)

# Re-export: customer sampling
from .customer_sampling import (
    _normalize_end_month,
    _eligible_customer_mask_for_month,
    _participation_distinct_target,
    _sample_customers,
)

# Re-export: promotions
from .promotions import (
    apply_promotions,
)

# Re-export: pricing
from .pricing import (
    compute_prices,
)

# Re-export: month planning / allocation
from .allocation import (
    macro_month_weights,
    build_rows_per_month,
)

# Backward compatibility: PA_AVAILABLE was defined in the original core.py
try:
    import pyarrow as pa  # type: ignore
except Exception:
    pa = None

PA_AVAILABLE = pa is not None

__all__ = [
    # date logic
    "compute_dates",

    # order logic
    "build_month_demand",
    "build_orders",

    # customer sampling
    "_normalize_end_month",
    "_eligible_customer_mask_for_month",
    "_participation_distinct_target",
    "_sample_customers",

    # promotions
    "apply_promotions",

    # pricing
    "compute_prices",

    # month planning
    "macro_month_weights",
    "build_rows_per_month",
]
