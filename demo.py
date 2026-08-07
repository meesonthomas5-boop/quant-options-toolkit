# demo.py
#
# End-to-end demonstration of the options_pricing_toolkit.
#
# Workflow:
#   1. Define a standard ATM European call option.
#   2. Price it with Black-Scholes, Bachelier, and Monte Carlo.
#   3. Compute Black-Scholes Greeks via finite differences.
#   4. Recover implied volatility from the Black-Scholes price.
#   5. Generate and save two diagnostic plots to the docs/ directory.

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Path setup — allows the script to be run from any working directory
# without installing the package.  Adds the project's src/ folder to the
# module search path so that `import options_pricing_toolkit` resolves
# correctly regardless of where Python is invoked from.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]   # Project root (two levels above this file)
sys.path.append(str(ROOT / "src"))           # Expose src/ to the Python import system

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


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

# Create docs/ at the project root if it does not already exist.
# All generated plot images will be saved here.
docs_dir = ROOT / "docs"
docs_dir.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Option specification
# ---------------------------------------------------------------------------

# Standard at-the-money (ATM) European call:
#   Spot = Strike = 100, 1-year maturity, 5% risk-free rate, 20% vol
spec = OptionSpec(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    volatility=0.2,
    option_type="call",
)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

bs   = black_scholes_price(spec)                       # Analytical BSM price
bach = bachelier_price(spec)                           # Analytical Bachelier price
mc, se = monte_carlo_price(spec, n_paths=100_000, seed=7)  # MC price + standard error


# ---------------------------------------------------------------------------
# Greeks and implied volatility
# ---------------------------------------------------------------------------

greeks = compute_greeks(spec)                 # Delta, Vega, Rho, Theta via finite differences
iv     = implied_volatility_from_price(bs, spec)  # Recover σ from the BSM price (should ≈ 0.20)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

print(f"Black-Scholes call price: {bs:.4f}")
print(f"Bachelier call price:     {bach:.4f}")
print(f"Monte Carlo call price:   {mc:.4f} ± {se:.4f}")   # ± one standard error

print("Greeks:")
for name, value in greeks.items():
    print(f"  {name}: {value:.4f}")

print(f"Implied volatility from BS price: {iv:.4f}")       # Round-trip check; expect 0.2000


# ---------------------------------------------------------------------------
# Volatility sensitivity plot
# ---------------------------------------------------------------------------

# Two separate vol grids because the models use different vol conventions:
#   Black-Scholes uses lognormal vol (dimensionless, e.g. 0.05–0.60)
#   Bachelier uses normal vol (same units as the asset price, e.g. 1–30)
bs_vol_grid   = np.linspace(0.05, 0.60, 40)   # 40 lognormal vol levels from 5% to 60%
bach_vol_grid = np.linspace(1.0,  30.0, 40)   # 40 normal vol levels from 1 to 30

plot_volatility_sensitivity(
    spec,
    bs_vol_grid,
    bach_vol_grid,
    save_path=str(docs_dir / "volatility_sensitivity.png"),
    show=True,   # Display interactively; also saved to disk
)


# ---------------------------------------------------------------------------
# Terminal price distribution plot
# ---------------------------------------------------------------------------

# Simulate 30k GBM paths and plot the resulting lognormal distribution.
# The strike is overlaid as a dashed red line so the in-the-money region
# is immediately visible.
plot_terminal_distribution(
    spec,
    n_paths=30_000,
    save_path=str(docs_dir / "terminal_distribution.png"),
    show=True,
)


# ---------------------------------------------------------------------------
# Confirm saved output paths
# ---------------------------------------------------------------------------

print("Saved plots:")
print(f"  {docs_dir / 'volatility_sensitivity.png'}")
print(f"  {docs_dir / 'terminal_distribution.png'}")
