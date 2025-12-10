import numpy as np
from src.facts.sales.sales_logic.globals import State

def compute_prices(rng, n, unit_price, unit_cost, promo_pct):

    promo_disc = unit_price * (promo_pct / 100.0)

    rnd_pct = rng.choice([0, 5, 10, 15, 20], n, p=[0.85, 0.06, 0.04, 0.03, 0.02])
    rnd_disc = unit_price * (rnd_pct * 0.01)

    discount_amt = np.maximum(promo_disc, rnd_disc)
    discount_amt *= rng.choice([0.90, 0.95, 1.00, 1.05, 1.10], n)
    discount_amt = np.round(discount_amt * 4) / 4
    discount_amt = np.minimum(discount_amt, unit_price - 0.01)

    # -----------------------------
    # PRICING MODE & FACTOR LOGIC
    # -----------------------------
    mode = getattr(State, "pricing_mode", "random")
    discrete = getattr(State, "discrete_factors", False)
    print("PRICE LOGIC: mode=", mode, "discrete_flag=", discrete)

    if mode == "discrete" or discrete:
        factor = rng.choice([0.02, 0.03, 0.04], size=n)
    elif mode == "bucketed":
        factor = rng.uniform(0.02, 0.04, size=n)
    else:
        factor = rng.uniform(0.02, 0.04, size=n)

    # -----------------------------
    # DISCRETE MODE BUCKETING
    # -----------------------------
    if mode == "discrete" or discrete:

        # bucket discount amount
        disc_bucket = getattr(State, "discount_bucket_size", 0.50)
        discount_amt = np.round(discount_amt / disc_bucket) * disc_bucket

        # bucket unit price + cost
        unit_bucket = getattr(State, "unit_bucket_size", 1.00)
        unit_price = np.round(unit_price / unit_bucket) * unit_bucket
        unit_cost  = np.round(unit_cost  / unit_bucket) * unit_bucket

        print(
            ">>> DISCRETE BLOCK RUNNING (unit_bucket=%s, disc_bucket=%s) <<<"
            % (getattr(State, "unit_bucket_size", None), getattr(State, "discount_bucket_size", None))
        )

    # -----------------------------
    # BASE PRICE CALCULATION
    # -----------------------------
    final_unit_price = np.round(unit_price * factor, 2)
    final_unit_cost  = np.round(unit_cost  * factor, 2)
    final_discount_amt = np.round(discount_amt * factor, 2)

    enforce = getattr(State, "enforce_min_price", False)
    if enforce:
        final_unit_price = np.clip(final_unit_price, 1.00, None)
        final_unit_cost  = np.clip(final_unit_cost, 0.50, None)
    # note: do not clip net here yet (we'll compute it from components)

    # -----------------------------
    # OPTION 2 — make unit price & discount share the same grid
    # -----------------------------
    if mode == "discrete" or discrete:
        unit_bucket = getattr(State, "unit_bucket_size", 1.00)

        # Bucket unit price on unit grid
        final_unit_price  = np.round(final_unit_price / unit_bucket) * unit_bucket

        # Bucket discount on its own grid (NOT the unit grid)
        disc_bucket = getattr(State, "discount_bucket_size", 0.50)
        final_discount_amt = np.round(final_discount_amt / disc_bucket) * disc_bucket

        # Compute net using the two bucketed values
        final_net_price = np.round(final_unit_price - final_discount_amt, 2)


        # enforce minimum net price after difference
        if enforce:
            final_net_price = np.clip(final_net_price, 1.00, None)
        else:
            final_net_price = np.clip(final_net_price, 0.01, None)

    # Bucketed mode (original behaviour) — keep as-is
    elif mode == "bucketed":
        bucket = getattr(State, "bucket_size", 0.25)
        final_unit_price = np.round(final_unit_price / bucket) * bucket
        final_unit_cost  = np.round(final_unit_cost  / bucket) * bucket
        final_discount_amt = np.round(final_discount_amt / bucket) * bucket
        final_net_price = np.round(final_unit_price - final_discount_amt, 2)
        if enforce:
            final_net_price = np.clip(final_net_price, 1.00, None)
        else:
            final_net_price = np.clip(final_net_price, 0.01, None)

    else:
        # random/default: keep net as computed earlier
        if 'final_net_price' not in locals():
            final_net_price = np.round(final_unit_price - final_discount_amt, 2)
            if enforce:
                final_net_price = np.clip(final_net_price, 1.00, None)
            else:
                final_net_price = np.clip(final_net_price, 0.01, None)


    return {
        "discount_amt": final_discount_amt,
        "final_unit_price": final_unit_price,
        "final_unit_cost": final_unit_cost,
        "final_net_price": final_net_price,
    }
