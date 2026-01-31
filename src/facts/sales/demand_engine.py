import numpy as np
from dataclasses import dataclass

# -------------------------------------------------
# Regime configuration (top-level, deterministic)
# -------------------------------------------------
_REGIMES = {
    "expansion": {
        "drift": 0.004,
        "volatility": 0.015,
        "momentum": 0.30,
        "transition_prob": 0.03,
        "trend_reversion": 0.98,
        "next": {"expansion": 0.85, "plateau": 0.10, "compression": 0.05},
    },
    "plateau": {
        "drift": 0.000,
        "volatility": 0.008,
        "momentum": 0.15,
        "transition_prob": 0.05,
        "trend_reversion": 0.98,
        "next": {"plateau": 0.80, "expansion": 0.10, "compression": 0.10},
    },
    "compression": {
        "drift": -0.003,
        "volatility": 0.020,
        "momentum": 0.35,
        "transition_prob": 0.08,
        "trend_reversion": 0.97,
        "next": {"compression": 0.75, "plateau": 0.15, "shock": 0.10},
    },
    "shock": {
        "drift": -0.020,
        "volatility": 0.050,
        "momentum": 0.60,
        "transition_prob": 0.00,  # shocks resolve, not transition
        "trend_reversion": 0.95,
        "next": {"plateau": 0.60, "compression": 0.25, "expansion": 0.15},
    },
}

# -------------------------------------------------
# Signals emitted by the engine (consumed elsewhere)
# -------------------------------------------------
@dataclass(frozen=True)
class DemandSignals:
    demand_multiplier: float
    price_pressure: float
    promo_intensity: float
    customer_activity_bias: float


# -------------------------------------------------
# Demand engine (stateful, non-cyclical)
# -------------------------------------------------
class DemandEngine:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

        # latent state
        self.level = 1.0
        self.trend = 0.0
        self.momentum = 0.0
        self.regime = "plateau"
        self.shock_remaining = 0

    # -------------------------------------------------
    # Advance one time step
    # -------------------------------------------------
    def step(self) -> DemandSignals:
        cfg = _REGIMES[self.regime]

        # -------------------------------------------------
        # 1. Regime transition (asymmetric, regime-specific)
        # -------------------------------------------------
        if self.shock_remaining <= 0:
            if self.rng.random() < cfg["transition_prob"]:
                next_regs = list(cfg["next"].keys())
                probs = list(cfg["next"].values())
                self.regime = self.rng.choice(next_regs, p=probs)

                if self.regime == "shock":
                    self.shock_remaining = int(self.rng.integers(2, 6))

                cfg = _REGIMES[self.regime]

        # -------------------------------------------------
        # 2. Structural drift with mean reversion
        # -------------------------------------------------
        self.trend += cfg["drift"]
        self.trend *= cfg["trend_reversion"]

        # -------------------------------------------------
        # 3. Correlated noise (momentum)
        # -------------------------------------------------
        noise = self.rng.normal(0.0, cfg["volatility"])
        self.momentum = (
            cfg["momentum"] * self.momentum
            + (1.0 - cfg["momentum"]) * noise
        )

        # -------------------------------------------------
        # 4. Shock effect (severity depends on prior regime)
        # -------------------------------------------------
        shock_effect = 0.0
        if self.shock_remaining > 0:
            if self.regime in ("compression", "shock"):
                shock_mu, shock_sigma = -0.04, 0.03
            else:
                shock_mu, shock_sigma = -0.02, 0.015

            shock_effect = self.rng.normal(shock_mu, shock_sigma)
            self.shock_remaining -= 1

        # -------------------------------------------------
        # 5. Update latent demand level
        # -------------------------------------------------
        self.level += self.trend + self.momentum + shock_effect
        self.level = float(np.clip(self.level, 0.55, 1.80))

        # -------------------------------------------------
        # 6. Emit signals (consumed downstream)
        # -------------------------------------------------
        demand_multiplier = self.level

        price_pressure = float(
            np.clip(1.0 + 0.6 * (self.level - 1.0), 0.90, 1.20)
        )

        promo_intensity = float(
            np.clip(1.15 - self.level, 0.0, 0.6)
        )

        customer_activity_bias = float(
            np.clip(0.75 + 0.6 * self.level, 0.5, 1.6)
        )

        return DemandSignals(
            demand_multiplier=demand_multiplier,
            price_pressure=price_pressure,
            promo_intensity=promo_intensity,
            customer_activity_bias=customer_activity_bias,
        )


# -------------------------------------------------
# Timeline builder (one step per unique time bucket)
# -------------------------------------------------
def build_demand_timeline(rng: np.random.Generator, dates):
    """
    Build one DemandSignals entry per unique time bucket
    (month or week, depending on caller).

    Returns:
        dict[datetime64 -> DemandSignals]
    """
    engine = DemandEngine(rng)
    timeline = {}

    # Explicit ordering for determinism
    unique_dates = np.sort(np.unique(dates))

    for d in unique_dates:
        timeline[d] = engine.step()

    return timeline
