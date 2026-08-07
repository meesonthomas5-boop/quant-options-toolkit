from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np

from options_pricing_toolkit import (
    OptionSpec,
    bachelier_price,
    black_scholes_price,
    compute_greeks,
    implied_volatility_from_price,
    monte_carlo_price,
    plot_terminal_distribution,
    plot_volatility_sensitivity,
)


spec = OptionSpec(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.2,
    option_type="call",
)

bs = black_scholes_price(spec)
bach = bachelier_price(spec)
mc, se = monte_carlo_price(spec, n_paths=100_000, seed=7)
greeks = compute_greeks(spec)
iv = implied_volatility_from_price(bs, spec)

print(f"Black-Scholes call price: {bs:.4f}")
print(f"Bachelier call price: {bach:.4f}")
print(f"Monte Carlo call price: {mc:.4f} ± {se:.4f}")
print("Greeks:")
for name, value in greeks.items():
    print(f"  {name}: {value:.4f}")
print(f"Implied volatility from BS price: {iv:.4f}")

vol_grid = np.linspace(0.05, 0.60, 40)
plot_volatility_sensitivity(spec, vol_grid)
plot_terminal_distribution(spec, n_paths=30_000)
