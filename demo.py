from pathlib import Path
import sys

# __file__ is the absolute path to this script.
# .resolve() canonicalises it (resolves symlinks, makes it absolute).
# .parents[1] walks two levels up — from the script's folder to the project root.
ROOT = Path(__file__).resolve().parents[1]

# Add the project's src/ directory to Python's module search path at runtime.
# This lets us import options_pricing_toolkit without installing the package,
# which is handy during development when you're running scripts directly.
sys.path.append(str(ROOT / "src"))

import numpy as np

# Import everything we need from the toolkit — explicit imports make it clear
# exactly which parts of the module this demo depends on.
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


# Create the docs/ directory if it doesn't already exist.
# exist_ok=True means no error is raised if the folder is already there.
docs_dir = ROOT / "docs"
docs_dir.mkdir(exist_ok=True)

# Define a single at-the-money call option that all the demo calculations will use.
# Spot == Strike means the option is exactly at-the-money.
spec = OptionSpec(
    spot=100.0,
    strike=100.0,
    maturity=1.0,    # 1 year
    rate=0.05,       # 5% risk-free rate
    volatility=0.2,  # 20% vol
    option_type="call",
)

# --- Compute All Prices and Greeks Up Front ---
# We run every calculation first and store the results, then print them together.
# This keeps the output section clean and makes it easy to add more calculations later.

bs   = black_scholes_price(spec)
bach = bachelier_price(spec)
mc, se = monte_carlo_price(spec, n_paths=100_000, seed=7)  # se = standard error
greeks = compute_greeks(spec)

# Round-trip implied vol: pass the BS price back into the IV solver.
# The result should be very close to 0.20 (our input vol), confirming consistency.
iv = implied_volatility_from_price(bs, spec)

# --- Print Results ---
print(f"Black-Scholes call price: {bs:.4f}")
print(f"Bachelier call price: {bach:.4f}")
print(f"Monte Carlo call price: {mc:.4f} ± {se:.4f}")  # ± shows the sampling uncertainty
print("Greeks:")
for name, value in greeks.items():
    print(f"  {name}: {value:.4f}")   # Two-space indent to visually group them under the header
print(f"Implied volatility from BS price: {iv:.4f}")

# --- Volatility Grids for Sensitivity Plots ---

# 40 evenly-spaced lognormal vol values from 5% to 60% for the Black-Scholes panel.
# This range covers everything from low-vol blue-chip stocks to high-vol small caps.
bs_vol_grid = np.linspace(0.05, 0.60, 40)

# 40 evenly-spaced normal vol values from 1 to 30 for the Bachelier panel.
# Bachelier vol is denominated in the same units as the price (e.g. dollars),
# so the scale is completely different from the lognormal vol above.
bach_vol_grid = np.linspace(1.0, 30.0, 40)

# --- Generate and Save Plots ---

# Plot how each model's price changes as volatility increases.
# Passing save_path writes the figure to disk; show=True also opens it interactively.
plot_volatility_sensitivity(
    spec,
    bs_vol_grid,
    bach_vol_grid,
    save_path=str(docs_dir / "volatility_sensitivity.png"),
    show=True,
)

# Plot the histogram of simulated terminal prices from the GBM model.
# Using 30,000 paths here (vs 100,000 for pricing) — fewer paths is fine for
# a visual distribution check where we don't need high numerical precision.
plot_terminal_distribution(
    spec,
    n_paths=30_000,
    save_path=str(docs_dir / "terminal_distribution.png"),
    show=True,
)

# Confirm where the output files landed so the user can find them easily.
print("Saved plots:")
print(f"  {docs_dir / 'volatility_sensitivity.png'}")
print(f"  {docs_dir / 'terminal_distribution.png'}")
