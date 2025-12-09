import numpy as np

def compute_prices(rng, n, unit_price, unit_cost, promo_pct):
    promo_disc = unit_price * (promo_pct / 100.0)

    rnd_pct = rng.choice([0, 5, 10, 15, 20], n, p=[0.85, 0.06, 0.04, 0.03, 0.02])
    rnd_disc = unit_price * (rnd_pct * 0.01)

    discount_amt = np.maximum(promo_disc, rnd_disc)
    discount_amt *= rng.choice([0.90, 0.95, 1.00, 1.05, 1.10], n)
    discount_amt = np.round(discount_amt * 4) / 4
    discount_amt = np.minimum(discount_amt, unit_price - 0.01)

    # Final price transforms
    factor = rng.uniform(0.02, 0.04, size=n)
    final_unit_price = np.round(unit_price * factor, 2)
    final_unit_cost = np.round(unit_cost * factor, 2)
    final_discount_amt = np.round(discount_amt * factor, 2)
    final_net_price = np.round(final_unit_price - final_discount_amt, 2)
    final_net_price = np.clip(final_net_price, 0.01, None)

    return {
        "discount_amt": final_discount_amt,
        "final_unit_price": final_unit_price,
        "final_unit_cost": final_unit_cost,
        "final_net_price": final_net_price,
    }
