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
import pandas as pd

# Import everything we need from the toolkit — explicit imports make it clear
# exactly which parts of the module this demo depends on.
from options_pricing_toolkit import (
    OptionSpec,
    bachelier_price,
    benchmark_mc_vs_bs,
    black_scholes_price,
    compute_greek_profiles,
    compute_greeks,
    compute_iv_surface,
    detect_mispricing,
    implied_volatility_from_price,
    monte_carlo_price,
    plot_greek_profiles,
    plot_iv_surface,
    plot_mc_benchmark,
    plot_terminal_distribution,
    plot_volatility_sensitivity,
)

# Create the docs/ directory if it doesn't already exist.
# exist_ok=True means no error is raised if the folder is already there.
docs_dir = ROOT / "docs"
docs_dir.mkdir(exist_ok=True)

# Define a single at-the-money call option that all demo calculations will use.
# Spot == Strike means the option is exactly at-the-money.
spec = OptionSpec(
    spot=100.0,
    strike=100.0,
    maturity=1.0,   # 1 year to expiry
    rate=0.05,      # 5% continuously compounded risk-free rate
    volatility=0.2, # 20% annualised lognormal volatility
    option_type="call",
)

# ---------------------------------------------------------------------------
# Section 1 — Core Pricing
# Run all three pricers against the base spec and print results side by side
# so the model differences are immediately visible.
# ---------------------------------------------------------------------------

print("=" * 60)
print("SECTION 1 — CORE PRICING")
print("=" * 60)

bs         = black_scholes_price(spec)
bach       = bachelier_price(spec)
mc, se     = monte_carlo_price(spec, n_paths=100_000, seed=7)  # se = standard error
iv         = implied_volatility_from_price(bs, spec)

# Round-trip implied vol: feed the BS price back into the IV solver.
# The result should recover exactly 0.20, confirming the pricer and
# bisection solver are mutually consistent.
print(f"Black-Scholes price : {bs:.4f}")
print(f"Bachelier price     : {bach:.4f}")
print(f"Monte Carlo price   : {mc:.4f}  ±  {se:.4f}")  # ± quantifies sampling uncertainty
print(f"Implied volatility  : {iv:.4f}  (round-trip check, should equal 0.2000)")

# ---------------------------------------------------------------------------
# Section 2 — MC Benchmark Sweep
# Price across a grid of 40 strikes and measure how closely Monte Carlo
# tracks the Black-Scholes analytical price at each node.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 2 — MONTE CARLO BENCHMARK SWEEP")
print("=" * 60)

# 40 evenly-spaced strikes from deep ITM (70) to deep OTM (130).
# This range exercises the pricer across the full moneyness spectrum.
strikes_grid = np.linspace(70, 130, 40)

# 500k paths keeps sampling noise well below the 0.5% accuracy threshold
# across all nodes. The min_price_filter drops nodes where the BS price is
# below $0.05 — at those prices relative error is numerically unstable and
# financially meaningless, so including them would distort the benchmark.
benchmark_df = benchmark_mc_vs_bs(
    base_spec=spec,
    strikes=strikes_grid,
    n_paths=500_000,
    seed=42,
    min_price_filter=0.05,
)

# Split the results into nodes that cleared the price filter and those that didn't.
# Only valid nodes are used for the accuracy assertion below.
valid    = benchmark_df.dropna(subset=["rel_error_pct"])
excluded = benchmark_df[benchmark_df["rel_error_pct"].isna()]

max_rel_error       = valid["rel_error_pct"].max()
mean_rel_error      = valid["rel_error_pct"].mean()
pct_below_threshold = (valid["rel_error_pct"] < 0.5).mean() * 100

print(f"Strikes tested          : {len(benchmark_df)}")
print(f"Nodes included (≥$0.05) : {len(valid)}")
print(f"Nodes excluded (<$0.05) : {len(excluded)}  (deep OTM — relative error not meaningful)")
print(f"Mean relative error     : {mean_rel_error:.4f}%")
print(f"Max relative error      : {max_rel_error:.4f}%")
print(f"Nodes below 0.5% target : {pct_below_threshold:.1f}%")

# Hard assertion — if this fires, the MC engine has failed the accuracy target
# and something is wrong with the simulation or path count.
assert max_rel_error < 0.5, (
    f"MC accuracy target breached: max relative error = {max_rel_error:.4f}%"
)
print("✓ All valid nodes are within the 0.5% accuracy threshold.")

# Write the full per-node results to CSV so the benchmark is auditable
# without re-running the simulation.
benchmark_csv = docs_dir / "mc_benchmark.csv"
benchmark_df.to_csv(benchmark_csv, index=False)
print(f"  Saved: {benchmark_csv}")

# Two-panel plot: prices side by side on the left, relative error with the
# 0.5% threshold line on the right.
plot_mc_benchmark(
    benchmark_df,
    save_path=str(docs_dir / "mc_benchmark.png"),
    show=True,
)

# ---------------------------------------------------------------------------
# Section 3 — Implied Volatility Surface & Mispricing Detection
# Build a 2D IV surface over a strike × maturity grid, then flag any node
# where the implied vol deviates from the base vol beyond a set threshold.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 3 — IMPLIED VOLATILITY SURFACE & MISPRICING DETECTION")
print("=" * 60)

# 20 strikes × 10 maturities = 200 surface nodes.
# Maturities run from 3 months to 2 years, covering the short and medium term.
surface_strikes    = np.linspace(80, 120, 20)
surface_maturities = np.linspace(0.25, 2.0, 10)

iv_surface_df = compute_iv_surface(
    base_spec=spec,
    strikes=surface_strikes,
    maturities=surface_maturities,
)

# Flag nodes where the recovered IV deviates from the base vol (20%) by more
# than 5 percentage points. On a synthetic flat surface this produces zero flags.
# Substitute real market quotes into compute_iv_surface to generate live signals.
mispricing_df = detect_mispricing(
    iv_surface=iv_surface_df,
    base_iv=spec.volatility,
    threshold_pct=5.0,
)

n_mispriced = mispricing_df["mispriced"].sum()
print(f"Surface nodes computed : {len(iv_surface_df)}")
print(f"Mispriced nodes flagged: {n_mispriced}")
print(f"  (0 expected on a synthetic flat surface — substitute real quotes to find live signals)")

# Save both the raw surface and the annotated mispricing table.
iv_csv         = docs_dir / "iv_surface.csv"
mispricing_csv = docs_dir / "mispricing_flags.csv"
iv_surface_df.to_csv(iv_csv, index=False)
mispricing_df.to_csv(mispricing_csv, index=False)
print(f"  Saved: {iv_csv}")
print(f"  Saved: {mispricing_csv}")

# Contour plot of the surface; any flagged nodes are overlaid as red scatter points.
plot_iv_surface(
    iv_surface_df,
    mispricing_df=mispricing_df,
    save_path=str(docs_dir / "iv_surface.png"),
    show=True,
)

# ---------------------------------------------------------------------------
# Section 4 — Greek Profiles Across Strikes
# Compute delta, vega, rho, and theta at every strike in the benchmark grid
# and export the results as both a printed table and a structured CSV.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 4 — GREEK PROFILES ACROSS STRIKE GRID")
print("=" * 60)

# Reuse strikes_grid from Section 2 so the Greek profile covers the same
# moneyness range as the MC benchmark — makes cross-referencing straightforward.
greek_df = compute_greek_profiles(
    base_spec=spec,
    strikes=strikes_grid,
)

# to_string with a custom float formatter gives a clean fixed-width table
# without needing to import tabulate or any other formatting library.
print(greek_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# The CSV is the primary structured output for this section — a downstream
# backtesting engine, risk system, or report generator can read it directly
# without re-running any pricing code.
greek_csv = docs_dir / "greek_profiles.csv"
greek_df.to_csv(greek_csv, index=False)
print(f"\n  Saved: {greek_csv}")

# Four-panel figure: one subplot per Greek, all sharing the same strike x-axis
# so the relationships between them are easy to compare visually.
plot_greek_profiles(
    greek_df,
    save_path=str(docs_dir / "greek_profiles.png"),
    show=True,
)

# ---------------------------------------------------------------------------
# Section 5 — Volatility Sensitivity & Terminal Distribution
# These two plots existed in the original demo and are retained unchanged.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 5 — VOLATILITY SENSITIVITY & TERMINAL DISTRIBUTION")
print("=" * 60)

# 40 evenly-spaced lognormal vol values from 5% to 60% for the BS panel.
# This range covers low-vol blue-chip stocks through high-vol small caps.
bs_vol_grid = np.linspace(0.05, 0.60, 40)

# 40 evenly-spaced normal vol values from 1 to 30 for the Bachelier panel.
# Bachelier vol is in price units (e.g. dollars) so the scale is entirely
# different from the lognormal vol grid above — they are not comparable directly.
bach_vol_grid = np.linspace(1.0, 30.0, 40)

# Side-by-side price vs vol curves for both models.
plot_volatility_sensitivity(
    spec, bs_vol_grid, bach_vol_grid,
    save_path=str(docs_dir / "volatility_sensitivity.png"),
    show=True,
)

# Histogram of GBM terminal prices with the strike overlaid as a dashed line.
# 30,000 paths is enough for a clear visual distribution — we don't need the
# higher path count used for pricing since visual accuracy is the only goal here.
plot_terminal_distribution(
    spec, n_paths=30_000,
    save_path=str(docs_dir / "terminal_distribution.png"),
    show=True,
)

# ---------------------------------------------------------------------------
# Section 6 — Greeks for the Base Spec
# Single-point Greek calculation for the ATM spec, printed as a quick reference.
# The right-align format spec (>6) keeps the Greek names in a tidy column.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SECTION 6 — GREEKS FOR BASE SPEC")
print("=" * 60)

greeks = compute_greeks(spec)
for name, value in greeks.items():
    print(f"  {name:>6}: {value:.4f}")

# ---------------------------------------------------------------------------
# Final Output Summary
# Print every output file path so the user knows exactly where to find results
# without having to search the filesystem.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("ALL OUTPUTS SAVED TO:", docs_dir)
print("=" * 60)

# Enumerate every file the demo produces — CSVs for structured data,
# PNGs for figures. Listing them explicitly also serves as a checklist:
# if a file is missing from docs/ after a run, something went wrong upstream.
output_files = [
    "mc_benchmark.csv",
    "mc_benchmark.png",
    "iv_surface.csv",
    "mispricing_flags.csv",
    "iv_surface.png",
    "greek_profiles.csv",
    "greek_profiles.png",
    "volatility_sensitivity.png",
    "terminal_distribution.png",
]

for f in output_files:
    print(f"  {docs_dir / f}")
