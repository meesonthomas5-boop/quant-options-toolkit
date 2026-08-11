# Allow 'OptionSpec | None' style type hints in Python versions before 3.10
from __future__ import annotations

import math
from dataclasses import dataclass   # Gives us a clean way to define plain data-holding classes
from pathlib import Path             # Cross-platform file path handling
from typing import Callable          # Used to type-hint functions passed as arguments

import matplotlib.pyplot as plt
import numpy as np


# --- Probability Helpers ---
# These two functions implement the standard normal PDF and CDF manually.
# The Black-Scholes and Bachelier formulas both require evaluating these,
# so rather than pulling in scipy we define them using Python's built-in math module.

def normal_pdf(x: float) -> float:
    # The standard normal probability density function: (1 / sqrt(2π)) * e^(-x²/2)
    # Used in the Bachelier formula to compute the "density at the boundary" term.
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: float) -> float:
    # The standard normal cumulative distribution function.
    # math.erf is the error function; this identity converts it to the CDF exactly.
    # Used extensively in both Black-Scholes and Bachelier pricing.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --- Option Specification ---

@dataclass
class OptionSpec:
    # A dataclass is essentially a struct — Python auto-generates __init__, __repr__,
    # and __eq__ for us based on the fields declared below.
    spot: float        # Current price of the underlying asset (S)
    strike: float      # The price at which the option can be exercised (K)
    maturity: float    # Time to expiry in years (T)
    rate: float        # Continuously compounded risk-free interest rate (r)
    volatility: float  # Annualised volatility of the underlying (σ)
    option_type: str = "call"  # Either "call" or "put"; defaults to call

    def validate(self) -> None:
        # Every pricer calls this before doing any math, so bad inputs are caught
        # early with a clear message rather than producing a silent wrong answer.
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity < 0:
            # Maturity of exactly zero is allowed — it means the option has just expired.
            raise ValueError("maturity must be non-negative")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")


# --- Black-Scholes Pricer ---

def black_scholes_price(spec: OptionSpec) -> float:
    spec.validate()

    # Unpack fields into short local names to keep the formula lines readable
    S = spec.spot
    K = spec.strike
    T = spec.maturity
    r = spec.rate
    sigma = spec.volatility

    # At expiry the option is worth exactly its intrinsic value — no time value remains.
    # Handling this edge case avoids a division-by-zero in the d1/d2 calculation below.
    if T == 0:
        if spec.option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    # With zero volatility the asset price is deterministic, so we only need to check
    # whether the discounted forward price beats the strike — no optionality premium.
    if sigma == 0:
        forward_intrinsic = S - K * math.exp(-r * T)
        if spec.option_type == "call":
            return max(forward_intrinsic, 0.0)
        return max(-forward_intrinsic, 0.0)

    # Core Black-Scholes d1 and d2 terms.
    # d1 captures how far in-the-money the option is, adjusted for drift and vol.
    # d2 = d1 - σ√T is the risk-neutral probability of finishing in-the-money.
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Standard Black-Scholes closed-form prices.
    # For a call:  S·N(d1) − K·e^(−rT)·N(d2)
    # For a put:   K·e^(−rT)·N(−d2) − S·N(−d1)
    # N(−x) = 1 − N(x), so we just negate the argument rather than subtracting.
    if spec.option_type == "call":
        return S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
    return K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)


# --- Bachelier Pricer ---
# The Bachelier model assumes the underlying follows arithmetic (normal) Brownian motion
# rather than geometric Brownian motion. This means prices can go negative, which makes
# it more appropriate for interest rates or spreads than for equity prices.

def bachelier_price(spec: OptionSpec) -> float:
    spec.validate()

    S = spec.spot
    K = spec.strike
    T = spec.maturity
    r = spec.rate
    sigma = spec.volatility * S

    # Same expiry edge case as Black-Scholes — return intrinsic value immediately.
    if T == 0:
        if spec.option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    discount = math.exp(-r * T)          # Present-value factor e^(−rT)
    forward = S * math.exp(r * T)        # Risk-neutral forward price F = S·e^(rT)
    stdev = sigma * math.sqrt(T)         # Total normal vol over the life of the option

    if stdev == 0:
        # No randomness means the payoff is fully determined by the forward vs strike.
        call = discount * max(forward - K, 0.0)
    else:
        # d here plays the same conceptual role as d2 in Black-Scholes:
        # it measures how many standard deviations the forward is above the strike.
        d = (forward - K) / stdev

        # Bachelier call formula:  e^(−rT) · [(F−K)·N(d) + σ√T·n(d)]
        # The first term is the expected in-the-money payoff; the second is the
        # "edge" contribution from the probability density right at the strike.
        call = discount * ((forward - K) * normal_cdf(d) + stdev * normal_pdf(d))

    if spec.option_type == "call":
        return call

    # Put price via put-call parity in the Bachelier world:
    # Put = Call − e^(−rT)·(F − K)
    return call - discount * (forward - K)


# --- Monte Carlo Engine ---

def simulate_terminal_prices_gbm(
    spot: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_paths: int,
    seed: int | None = 42,   # Fixed seed by default so results are reproducible
) -> np.ndarray:
    # Build a seeded random number generator — preferred over np.random.seed()
    # because it is stateless and won't interfere with other RNG usage in the program.
    rng = np.random.default_rng(seed)

    # Draw n_paths independent standard-normal samples — one per simulated path.
    z = rng.standard_normal(n_paths)

    # Under GBM the log-return over [0, T] is normally distributed:
    #   ln(S_T / S_0) ~ N((r − σ²/2)·T,  σ²·T)
    # We split this into a deterministic drift and a stochastic diffusion term.
    drift = (rate - 0.5 * volatility * volatility) * maturity
    diffusion = volatility * math.sqrt(maturity) * z

    # Exponentiate to recover the terminal asset price for each path.
    return spot * np.exp(drift + diffusion)


def monte_carlo_price(
    spec: OptionSpec,
    n_paths: int = 100_000,
    seed: int | None = 42,
) -> tuple[float, float]:   # Returns (price estimate, standard error)
    spec.validate()

    # Expired option — no simulation needed, payoff is deterministic.
    if spec.maturity == 0:
        if spec.option_type == "call":
            payoff = max(spec.spot - spec.strike, 0.0)
        else:
            payoff = max(spec.strike - spec.spot, 0.0)
        return payoff, 0.0   # Standard error is zero because there is no randomness

    # Simulate where the asset price ends up at maturity across all paths.
    terminal = simulate_terminal_prices_gbm(
        spec.spot,
        spec.maturity,
        spec.rate,
        spec.volatility,
        n_paths,
        seed,
    )

    # Compute the payoff on each path. np.maximum is the vectorised version of max(),
    # operating element-wise across the entire array in one shot.
    if spec.option_type == "call":
        payoffs = np.maximum(terminal - spec.strike, 0.0)
    else:
        payoffs = np.maximum(spec.strike - terminal, 0.0)

    # Discount each payoff back to today and average across paths.
    # By the law of large numbers this converges to the true risk-neutral expectation.
    discounted = np.exp(-spec.rate * spec.maturity) * payoffs
    price = float(np.mean(discounted))

    # Standard error of the mean = sample std dev / sqrt(n).
    # ddof=1 applies Bessel's correction, giving an unbiased variance estimate.
    # This tells us how uncertain our price estimate is due to sampling noise.
    stderr = float(np.std(discounted, ddof=1) / math.sqrt(n_paths))

    return price, stderr


# --- Numerical Greeks via Finite Differences ---

def finite_difference_greek(
    pricer: Callable[[OptionSpec], float],  # Any pricing function that accepts an OptionSpec
    spec: OptionSpec,
    parameter: str,   # The name of the OptionSpec field we want to bump, e.g. "spot"
    bump: float,      # The size of the perturbation h
) -> float:
    if bump <= 0:
        raise ValueError("bump must be positive")

    # Create two independent copies of the spec so we can bump each direction
    # without mutating the original. vars() returns the dataclass fields as a dict.
    up = OptionSpec(**vars(spec))
    down = OptionSpec(**vars(spec))

    # Apply +h and −h bumps to the chosen parameter using Python's reflection API.
    # The max(..., 1e-8) on the downward bump prevents parameters like volatility
    # from going negative, which would fail validation.
    setattr(up, parameter, getattr(up, parameter) + bump)
    setattr(down, parameter, max(getattr(down, parameter) - bump, 1e-8))

    # Central difference: (f(x+h) − f(x−h)) / 2h
    # This is second-order accurate — the error shrinks as h², not just h.
    return (pricer(up) - pricer(down)) / (2.0 * bump)


def compute_greeks(spec: OptionSpec) -> dict[str, float]:
    # All four Greeks are computed via finite differences against the Black-Scholes pricer.
    return {
        # Delta: sensitivity of price to a $1 move in the spot price
        "delta": finite_difference_greek(black_scholes_price, spec, "spot", 1e-2),

        # Vega: sensitivity of price to a 0.01 (1%) move in volatility
        "vega": finite_difference_greek(black_scholes_price, spec, "volatility", 1e-4),

        # Rho: sensitivity of price to a 0.01 (1%) move in the risk-free rate
        "rho": finite_difference_greek(black_scholes_price, spec, "rate", 1e-4),

        # Theta: rate of price decay per unit of time. The finite difference gives
        # dPrice/dT (which is positive for long options), so we negate it to get
        # the conventional theta sign — the daily cost of holding the option.
        "theta": -finite_difference_greek(black_scholes_price, spec, "maturity", 1e-4),
    }


# --- Implied Volatility via Bisection ---

def implied_volatility_from_price(
    market_price: float,
    base_spec: OptionSpec,
    tol: float = 1e-8,    # Stop when the model price is within this distance of market_price
    max_iter: int = 200,  # Safety cap — bisection converges fast so 200 is very generous
) -> float:
    # Search bounds for volatility. 1e-8 is effectively zero; 5.0 is 500% vol,
    # which is extreme enough to bracket any realistic market price.
    low = 1e-8
    high = 5.0

    for _ in range(max_iter):
        # Test the midpoint of the current bracket
        mid = 0.5 * (low + high)

        # Build a fresh spec with the candidate volatility — we can't mutate base_spec
        # because the caller may still need it after this function returns.
        spec = OptionSpec(
            spot=base_spec.spot,
            strike=base_spec.strike,
            maturity=base_spec.maturity,
            rate=base_spec.rate,
            volatility=mid,
            option_type=base_spec.option_type,
        )
        price = black_scholes_price(spec)

        # Convergence check — if we're close enough, return immediately
        if abs(price - market_price) < tol:
            return mid

        # Black-Scholes price is monotonically increasing in volatility, so:
        # if the model underprices, the true vol must be higher → raise the lower bound
        # if the model overprices, the true vol must be lower  → lower the upper bound
        if price < market_price:
            low = mid
        else:
            high = mid

    # If we exhaust all iterations without converging, return the best midpoint we have.
    return 0.5 * (low + high)


# --- Plotting Utilities ---

def _finalize_plot(save_path: str | None = None, show: bool = True) -> None:
    # Tighten subplot spacing so labels and titles don't overlap
    plt.tight_layout()

    if save_path is not None:
        output = Path(save_path)
        # Create any missing parent directories automatically (like 'mkdir -p')
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=160)   # 160 dpi gives a crisp image without huge file size

    if show:
        plt.show()   # Opens the interactive window
    else:
        plt.close()  # Free memory without displaying — useful in automated/headless runs


def plot_volatility_sensitivity(
    spec: OptionSpec,
    bs_vol_grid: np.ndarray,    # Array of lognormal vols to sweep for the BS panel
    bach_vol_grid: np.ndarray,  # Array of normal vols to sweep for the Bachelier panel
    save_path: str | None = None,
    show: bool = True,
) -> None:
    # Price the option at every point in the Black-Scholes volatility grid.
    # We rebuild a fresh OptionSpec for each vol level rather than mutating the original.
    bs_prices = []
    for vol in bs_vol_grid:
        updated = OptionSpec(
            spot=spec.spot,
            strike=spec.strike,
            maturity=spec.maturity,
            rate=spec.rate,
            volatility=float(vol),   # Cast from numpy scalar to plain float for safety
            option_type=spec.option_type,
        )
        bs_prices.append(black_scholes_price(updated))

    # Same sweep for the Bachelier model over its own vol grid.
    # Note: Bachelier vol is in price units (e.g. dollars), not a percentage,
    # so the grid range is very different from the BS one.
    bach_prices = []
    for vol in bach_vol_grid:
        updated = OptionSpec(
            spot=spec.spot,
            strike=spec.strike,
            maturity=spec.maturity,
            rate=spec.rate,
            volatility=float(vol),
            option_type=spec.option_type,
        )
        bach_prices.append(bachelier_price(updated))

    # Side-by-side subplots — one panel per model
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(bs_vol_grid, bs_prices, linewidth=2, label="Black-Scholes")
    axes[0].set_title("Black-Scholes Sensitivity")
    axes[0].set_xlabel("Lognormal Volatility")
    axes[0].set_ylabel("Option Price")
    axes[0].grid(alpha=0.3)   # Faint grid lines — visible but not distracting
    axes[0].legend()

    axes[1].plot(bach_vol_grid, bach_prices, linewidth=2, color="orange", label="Bachelier")
    axes[1].set_title("Bachelier Sensitivity")
    axes[1].set_xlabel("Normal Volatility")
    axes[1].set_ylabel("Option Price")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # Overall figure title that includes whether we're looking at calls or puts
    plt.suptitle(f"Volatility Sensitivity, {spec.option_type.capitalize()} Option")
    _finalize_plot(save_path=save_path, show=show)


def plot_terminal_distribution(
    spec: OptionSpec,
    n_paths: int = 50_000,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    # Run a GBM simulation to get the distribution of terminal asset prices
    terminal = simulate_terminal_prices_gbm(
        spot=spec.spot,
        maturity=spec.maturity,
        rate=spec.rate,
        volatility=spec.volatility,
        n_paths=n_paths,
    )

    plt.figure(figsize=(8, 5))
    # 60 bins gives enough resolution to see the lognormal shape clearly
    plt.hist(terminal, bins=60, alpha=0.8, edgecolor="black")

    # Vertical dashed line at the strike — makes it visually obvious how much of
    # the distribution sits in-the-money vs out-of-the-money
    plt.axvline(spec.strike, color="red", linestyle="--", label="Strike")
    plt.xlabel("Terminal Price")
    plt.ylabel("Frequency")
    plt.title("Simulated Terminal Price Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    _finalize_plot(save_path=save_path, show=show)


# --- Quick Smoke Test ---
# This block only runs when the file is executed directly (python options_pricing_toolkit.py),
# not when it is imported as a module by another script.

if __name__ == "__main__":
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,    # 1 year to expiry
        rate=0.05,       # 5% risk-free rate
        volatility=0.2,  # 20% annualised vol — a typical equity assumption
        option_type="call",
    )

    print("Black-Scholes:", round(black_scholes_price(spec), 4))
    print("Bachelier:", round(bachelier_price(spec), 4))

    # monte_carlo_price returns a (price, standard_error) tuple, so we unpack both
    mc_price, mc_err = monte_carlo_price(spec, n_paths=100_000, seed=7)
    print("Monte Carlo:", round(mc_price, 4), "+/-", round(mc_err, 4))

    # Print each Greek on its own line
    greeks = compute_greeks(spec)
    for name, value in greeks.items():
        print(name, round(value, 4))

    # Round-trip check: feed the BS price back in and recover the original vol.
    # If implied_volatility_from_price is working correctly, this should print ~0.2.
    implied = implied_volatility_from_price(black_scholes_price(spec), spec)
    print("Implied vol:", round(implied, 4))
