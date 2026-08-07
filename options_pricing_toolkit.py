# options_pricing_toolkit.py
#
# A self-contained toolkit for pricing European options using three models:
#   - Black-Scholes (lognormal, analytical)
#   - Bachelier     (normal, analytical)
#   - Monte Carlo   (GBM simulation, numerical)
#
# Also provides:
#   - Finite-difference Greek estimation
#   - Implied-volatility solver (bisection)
#   - Matplotlib visualisation helpers

from __future__ import annotations  # Enables postponed evaluation of type hints (PEP 563)

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Mathematical helpers
# ---------------------------------------------------------------------------

def normal_pdf(x: float) -> float:
    """
    Standard normal probability density function (PDF).

    Computes:
        φ(x) = exp(-x² / 2) / √(2π)

    Used in the Bachelier pricing formula where the PDF of the standard
    normal distribution appears directly in the analytical solution.
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function (CDF).

    Computes:
        Φ(x) = 0.5 * (1 + erf(x / √2))

    Delegates to math.erf, which is accurate to machine precision.
    Used in both the Black-Scholes and Bachelier closed-form formulas.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Option specification
# ---------------------------------------------------------------------------

@dataclass
class OptionSpec:
    """
    Immutable-by-convention container for all parameters that define a
    single European option contract.

    Attributes
    ----------
    spot       : Current price of the underlying asset (S₀).
    strike     : Strike / exercise price of the option (K).
    maturity   : Time to expiry in years (T).
    rate       : Continuously compounded risk-free interest rate (r).
    volatility : Annualised volatility of the underlying (σ).
                 Interpreted as *lognormal* vol for Black-Scholes and
                 *normal* vol for Bachelier.
    option_type: 'call' (right to buy) or 'put' (right to sell).
    """

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    option_type: str = "call"

    def validate(self) -> None:
        """
        Raise ValueError for any parameter combination that would produce
        undefined or economically meaningless results.

        Called at the start of every pricing function so that errors are
        caught early with a clear message rather than producing a silent
        NaN or an obscure math-domain error deep in the calculation.
        """
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity < 0:
            raise ValueError("maturity must be non-negative")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")


# ---------------------------------------------------------------------------
# Analytical pricers
# ---------------------------------------------------------------------------

def black_scholes_price(spec: OptionSpec) -> float:
    """
    Price a European option under the Black-Scholes-Merton (BSM) model.

    The BSM model assumes the underlying follows Geometric Brownian Motion
    (GBM), so the terminal price is lognormally distributed.

    Closed-form prices:
        Call: C = S·Φ(d₁) − K·e^(−rT)·Φ(d₂)
        Put:  P = K·e^(−rT)·Φ(−d₂) − S·Φ(−d₁)

    where:
        d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ√T)
        d₂ = d₁ − σ√T

    Edge cases handled explicitly:
        - T = 0  → immediate exercise value (intrinsic value).
        - σ = 0  → deterministic forward; price is discounted intrinsic.

    Parameters
    ----------
    spec : OptionSpec
        Fully populated option specification.

    Returns
    -------
    float
        Fair value of the option in the same currency as spot/strike.
    """
    spec.validate()

    # Unpack for readability
    S = spec.spot
    K = spec.strike
    T = spec.maturity
    r = spec.rate
    sigma = spec.volatility

    # --- Edge case: option has already expired ---
    # At expiry the option is worth its intrinsic value only.
    if T == 0:
        if spec.option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    # --- Edge case: zero volatility ---
    # With no randomness the asset grows deterministically at rate r,
    # so the forward price is S·e^(rT) and we discount back to today.
    if sigma == 0:
        forward_intrinsic = S - K * math.exp(-r * T)
        if spec.option_type == "call":
            return max(forward_intrinsic, 0.0)
        return max(-forward_intrinsic, 0.0)

    # --- Standard BSM formula ---
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if spec.option_type == "call":
        return S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
    # Put price derived from the same formula via put-call symmetry
    return K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)


def bachelier_price(spec: OptionSpec) -> float:
    """
    Price a European option under the Bachelier (normal) model.

    Unlike Black-Scholes, Bachelier assumes the *absolute* change in the
    underlying is normally distributed, making it suitable for assets that
    can go negative (e.g. interest rates, spreads).

    Closed-form prices (in terms of the forward F = S·e^(rT)):
        Call: C = e^(−rT) · [(F−K)·Φ(d) + σ√T·φ(d)]
        Put:  P = C − e^(−rT)·(F−K)          ← via put-call parity

    where:
        d = (F − K) / (σ√T)

    Edge cases handled:
        - T = 0  → intrinsic value.
        - σ√T = 0 → deterministic forward; discounted intrinsic.

    Parameters
    ----------
    spec : OptionSpec
        Fully populated option specification.
        `volatility` is interpreted as the *normal* (absolute) volatility.

    Returns
    -------
    float
        Fair value of the option.
    """
    spec.validate()

    S = spec.spot
    K = spec.strike
    T = spec.maturity
    r = spec.rate
    sigma = spec.volatility

    # --- Edge case: option has already expired ---
    if T == 0:
        if spec.option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    discount = math.exp(-r * T)          # Present-value discount factor e^(−rT)
    forward  = S * math.exp(r * T)       # Risk-neutral forward price F = S·e^(rT)
    stdev    = sigma * math.sqrt(T)      # Standard deviation of the terminal price distribution

    # --- Edge case: zero vol or zero time → deterministic outcome ---
    if stdev == 0:
        call = discount * max(forward - K, 0.0)
    else:
        d    = (forward - K) / stdev                                          # Standardised moneyness
        call = discount * ((forward - K) * normal_cdf(d) + stdev * normal_pdf(d))

    if spec.option_type == "call":
        return call
    # Bachelier put-call parity: P = C − discount·(F − K)
    return call - discount * (forward - K)


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def simulate_terminal_prices_gbm(
    spot: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_paths: int,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate terminal asset prices under Geometric Brownian Motion (GBM).

    Under GBM the exact solution for the terminal price is:
        S_T = S₀ · exp[(r − σ²/2)·T + σ·√T·Z]

    where Z ~ N(0,1). This avoids time-stepping discretisation error
    entirely because the log-price is normally distributed.

    Parameters
    ----------
    spot       : Initial asset price S₀.
    maturity   : Time horizon T (years).
    rate       : Continuously compounded risk-free rate r.
    volatility : Lognormal volatility σ.
    n_paths    : Number of independent simulation paths.
    seed       : RNG seed for reproducibility; pass None for a random seed.

    Returns
    -------
    np.ndarray of shape (n_paths,)
        Simulated terminal prices S_T for each path.
    """
    rng = np.random.default_rng(seed)          # Reproducible random number generator
    z   = rng.standard_normal(n_paths)         # Draw n_paths standard normal samples

    # Decompose the log-return into its deterministic drift and stochastic diffusion
    drift     = (rate - 0.5 * volatility * volatility) * maturity   # (r − σ²/2)·T
    diffusion = volatility * math.sqrt(maturity) * z                # σ·√T·Z

    return spot * np.exp(drift + diffusion)


def monte_carlo_price(
    spec: OptionSpec,
    n_paths: int = 100_000,
    seed: int | None = 42,
) -> tuple[float, float]:
    """
    Price a European option via Monte Carlo simulation under GBM.

    Procedure:
        1. Simulate n_paths terminal prices using simulate_terminal_prices_gbm.
        2. Compute the option payoff on each path.
        3. Discount payoffs back to today and average them.

    The standard error of the mean is also returned as a measure of
    simulation uncertainty; it shrinks as O(1/√n_paths).

    Parameters
    ----------
    spec    : OptionSpec  — option parameters.
    n_paths : int         — number of Monte Carlo paths (more → lower error).
    seed    : int | None  — RNG seed for reproducibility.

    Returns
    -------
    (price, stderr) : tuple[float, float]
        price  — Monte Carlo estimate of the option fair value.
        stderr — Standard error of the Monte Carlo estimate.
    """
    spec.validate()

    # --- Edge case: expired option → return deterministic intrinsic value ---
    if spec.maturity == 0:
        if spec.option_type == "call":
            payoff = max(spec.spot - spec.strike, 0.0)
        else:
            payoff = max(spec.strike - spec.spot, 0.0)
        return payoff, 0.0   # Zero standard error: no randomness at expiry

    # Step 1 — Simulate terminal prices
    terminal = simulate_terminal_prices_gbm(
        spec.spot,
        spec.maturity,
        spec.rate,
        spec.volatility,
        n_paths,
        seed,
    )

    # Step 2 — Compute per-path payoffs
    # Call payoff: max(S_T − K, 0);  Put payoff: max(K − S_T, 0)
    if spec.option_type == "call":
        payoffs = np.maximum(terminal - spec.strike, 0.0)
    else:
        payoffs = np.maximum(spec.strike - terminal, 0.0)

    # Step 3 — Discount to present value and compute the mean (the MC price)
    discounted = np.exp(-spec.rate * spec.maturity) * payoffs

    price  = float(np.mean(discounted))
    # Standard error = sample std dev / √n  (ddof=1 for unbiased std dev)
    stderr = float(np.std(discounted, ddof=1) / math.sqrt(n_paths))

    return price, stderr


# ---------------------------------------------------------------------------
# Greeks via finite differences
# ---------------------------------------------------------------------------

def finite_difference_greek(
    pricer: Callable[[OptionSpec], float],
    spec: OptionSpec,
    parameter: str,
    bump: float,
) -> float:
    """
    Estimate a first-order option Greek using a central finite difference.

    The central difference approximation is:
        ∂V/∂θ ≈ [V(θ + h) − V(θ − h)] / (2h)

    Central differences are second-order accurate — the error is O(h²) —
    which is significantly better than the O(h) error of one-sided schemes.

    Parameters
    ----------
    pricer    : Any function that accepts an OptionSpec and returns a price.
                This allows the same helper to be used with any model.
    spec      : Base OptionSpec from which bumped copies are derived.
    parameter : Name of the OptionSpec attribute to bump (e.g. 'spot').
    bump      : Half-width of the finite-difference interval h.
                Must be positive; a typical value is 1 % of the parameter.

    Returns
    -------
    float
        Numerical approximation of ∂V/∂parameter.
    """
    if bump <= 0:
        raise ValueError("bump must be positive")

    # Create independent copies so the original spec is never mutated
    up   = OptionSpec(**vars(spec))
    down = OptionSpec(**vars(spec))

    # Apply the symmetric bump; clamp the downward bump to a small positive
    # floor (1e-8) to prevent non-positive spot/strike/vol/maturity values
    setattr(up,   parameter, getattr(up,   parameter) + bump)
    setattr(down, parameter, max(getattr(down, parameter) - bump, 1e-8))

    return (pricer(up) - pricer(down)) / (2.0 * bump)


def compute_greeks(spec: OptionSpec) -> dict[str, float]:
    """
    Compute the four primary Black-Scholes Greeks via finite differences.

    Greek definitions and the parameter bumped for each:
        Delta (Δ) — ∂V/∂S        : sensitivity to the underlying price.
        Vega  (ν) — ∂V/∂σ        : sensitivity to volatility.
        Rho   (ρ) — ∂V/∂r        : sensitivity to the risk-free rate.
        Theta (Θ) — −∂V/∂T       : time decay (negative sign because as
                                    calendar time advances, T decreases).

    All Greeks are computed using the Black-Scholes pricer.

    Parameters
    ----------
    spec : OptionSpec — option parameters.

    Returns
    -------
    dict[str, float]
        Keys: 'delta', 'vega', 'rho', 'theta'.
    """
    return {
        # Bump spot by 1 cent (0.01) — small relative to typical spot prices
        "delta": finite_difference_greek(black_scholes_price, spec, "spot",       1e-2),
        # Bump vol by 0.01 % (0.0001) — small relative to typical vol levels
        "vega":  finite_difference_greek(black_scholes_price, spec, "volatility", 1e-4),
        # Bump rate by 0.01 % (0.0001 in decimal)
        "rho":   finite_difference_greek(black_scholes_price, spec, "rate",       1e-4),
        # Theta: negate because we bump maturity (increasing T) but theta
        # conventionally measures value lost as time *passes* (T decreases)
        "theta": -finite_difference_greek(black_scholes_price, spec, "maturity",  1e-4),
    }


# ---------------------------------------------------------------------------
# Implied volatility solver
# ---------------------------------------------------------------------------

def implied_volatility_from_price(
    market_price: float,
    base_spec: OptionSpec,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """
    Recover the implied volatility (IV) consistent with an observed market
    price using the Black-Scholes model and bisection search.

    Implied volatility is the value of σ that satisfies:
        BSM_price(σ) = market_price

    Bisection is chosen for its guaranteed convergence on a monotone
    function — BSM price is strictly increasing in σ — at the cost of
    being slower than Newton-Raphson. With max_iter=200 the interval
    [1e-8, 5.0] is narrowed to a width of ~5 / 2^200 ≈ 0, far below
    the default tolerance.

    Parameters
    ----------
    market_price : Observed option price to invert.
    base_spec    : OptionSpec with all parameters except volatility fixed.
                   The volatility field is ignored; it is solved for.
    tol          : Convergence tolerance on the price residual.
    max_iter     : Maximum number of bisection iterations.

    Returns
    -------
    float
        Implied volatility σ such that BSM_price(σ) ≈ market_price.
    """
    # Search bounds: [near-zero vol, 500% vol]
    # BSM price is monotonically increasing in σ, so bisection is valid
    low  = 1e-8
    high = 5.0

    for _ in range(max_iter):
        mid = 0.5 * (low + high)   # Candidate volatility (midpoint of current interval)

        # Evaluate BSM price at the candidate volatility
        spec = OptionSpec(
            spot=base_spec.spot,
            strike=base_spec.strike,
            maturity=base_spec.maturity,
            rate=base_spec.rate,
            volatility=mid,
            option_type=base_spec.option_type,
        )
        price = black_scholes_price(spec)

        # Check convergence: stop if the price residual is within tolerance
        if abs(price - market_price) < tol:
            return mid

        # Narrow the search interval based on whether the model price is
        # too low (need higher vol) or too high (need lower vol)
        if price < market_price:
            low = mid
        else:
            high = mid

    # Return the best midpoint estimate if max iterations are exhausted
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _finalize_plot(save_path: str | None = None, show: bool = True) -> None:
    """
    Apply final layout adjustments and either save and/or display the
    current Matplotlib figure, then clean up.

    Separating this logic avoids duplicating the same save/show/close
    boilerplate in every individual plotting function.

    Parameters
    ----------
    save_path : File path to save the figure to (PNG, PDF, etc.).
                Parent directories are created automatically if needed.
                Pass None to skip saving.
    show      : If True, display the figure interactively via plt.show().
                If False, close the figure silently (useful in batch runs
                or automated testing where no display is available).
    """
    plt.tight_layout()   # Adjust subplot spacing to prevent label overlap

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)   # Ensure output directory exists
        plt.savefig(output, dpi=160)                        # Save at 160 DPI for crisp output

    if show:
        plt.show()
    else:
        plt.close()   # Release memory when not displaying interactively


def plot_volatility_sensitivity(
    spec: OptionSpec,
    bs_vol_grid: np.ndarray,
    bach_vol_grid: np.ndarray,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot option price as a function of volatility for both the
    Black-Scholes and Bachelier models side-by-side.

    Useful for comparing how each model's price responds to changes in
    its respective volatility parameter (lognormal vs. normal vol).

    Parameters
    ----------
    spec          : Base OptionSpec (spot, strike, maturity, rate, type).
                    The volatility field is overridden by the grids below.
    bs_vol_grid   : Array of lognormal volatility values for Black-Scholes
                    (e.g. np.linspace(0.05, 0.60, 40)).
    bach_vol_grid : Array of normal volatility values for Bachelier
                    (e.g. np.linspace(1.0, 30.0, 40)).
    save_path     : Optional file path to save the figure.
    show          : Whether to display the figure interactively.
    """
    # --- Compute Black-Scholes prices across the lognormal vol grid ---
    bs_prices = []
    for vol in bs_vol_grid:
        updated = OptionSpec(
            spot=spec.spot,
            strike=spec.strike,
            maturity=spec.maturity,
            rate=spec.rate,
            volatility=float(vol),    # Override only the volatility
            option_type=spec.option_type,
        )
        bs_prices.append(black_scholes_price(updated))

    # --- Compute Bachelier prices across the normal vol grid ---
    bach_prices = []
    for vol in bach_vol_grid:
        updated = OptionSpec(
            spot=spec.spot,
            strike=spec.strike,
            maturity=spec.maturity,
            rate=spec.rate,
            volatility=float(vol),    # Override only the volatility
            option_type=spec.option_type,
        )
        bach_prices.append(bachelier_price(updated))

    # --- Build a 1×2 figure: Black-Scholes on the left, Bachelier on the right ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(bs_vol_grid, bs_prices, linewidth=2, label="Black-Scholes")
    axes[0].set_title("Black-Scholes Sensitivity")
    axes[0].set_xlabel("Lognormal Volatility")
    axes[0].set_ylabel("Option Price")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(bach_vol_grid, bach_prices, linewidth=2, color="orange", label="Bachelier")
    axes[1].set_title("Bachelier Sensitivity")
    axes[1].set_xlabel("Normal Volatility")
    axes[1].set_ylabel("Option Price")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.suptitle(f"Volatility Sensitivity, {spec.option_type.capitalize()} Option")
    _finalize_plot(save_path=save_path, show=show)


def plot_terminal_distribution(
    spec: OptionSpec,
    n_paths: int = 50_000,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot a histogram of simulated terminal asset prices under GBM.

    Visualises the lognormal distribution that underpins the Black-Scholes
    model and marks the strike price for reference, making it easy to see
    the proportion of paths that expire in-the-money.

    Parameters
    ----------
    spec      : OptionSpec — parameters used for the simulation.
    n_paths   : Number of GBM paths to simulate (more → smoother histogram).
    save_path : Optional file path to save the figure.
    show      : Whether to display the figure interactively.
    """
    # Simulate terminal prices using the GBM exact solution
    terminal = simulate_terminal_prices_gbm(
        spot=spec.spot,
        maturity=spec.maturity,
        rate=spec.rate,
        volatility=spec.volatility,
        n_paths=n_paths,
    )

    plt.figure(figsize=(8, 5))
    plt.hist(terminal, bins=60, alpha=0.8, edgecolor="black")   # 60 bins for smooth resolution
    plt.axvline(spec.strike, color="red", linestyle="--", label="Strike")  # Mark the strike
    plt.xlabel("Terminal Price")
    plt.ylabel("Frequency")
    plt.title("Simulated Terminal Price Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    _finalize_plot(save_path=save_path, show=show)


# ---------------------------------------------------------------------------
# Quick self-test / usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Define a standard at-the-money (ATM) European call option:
    #   S = K = 100, T = 1 year, r = 5%, σ = 20%
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        option_type="call",
    )

    # --- Analytical prices ---
    print("Black-Scholes:", round(black_scholes_price(spec), 4))
    print("Bachelier:",     round(bachelier_price(spec),     4))

    # --- Monte Carlo price with 100k paths and a fixed seed ---
    mc_price, mc_err = monte_carlo_price(spec, n_paths=100_000, seed=7)
    print("Monte Carlo:", round(mc_price, 4), "+/-", round(mc_err, 4))

    # --- Finite-difference Greeks ---
    greeks = compute_greeks(spec)
    for name, value in greeks.items():
        print(name, round(value, 4))

    # --- Round-trip implied vol: should recover 0.20 exactly ---
    implied = implied_volatility_from_price(black_scholes_price(spec), spec)
    print("Implied vol:", round(implied, 4))
