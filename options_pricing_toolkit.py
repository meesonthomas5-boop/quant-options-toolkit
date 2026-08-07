from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class OptionSpec:
    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    option_type: str = "call"

    def validate(self) -> None:
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


def black_scholes_price(spec: OptionSpec) -> float:
    spec.validate()
    S = spec.spot
    K = spec.strike
    T = spec.maturity
    r = spec.rate
    sigma = spec.volatility

    if T == 0:
        if spec.option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    if sigma == 0:
        forward_intrinsic = S - K * math.exp(-r * T)
        if spec.option_type == "call":
            return max(forward_intrinsic, 0.0)
        return max(-forward_intrinsic, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if spec.option_type == "call":
        return S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
    return K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)


def bachelier_price(spec: OptionSpec) -> float:
    spec.validate()
    S = spec.spot
    K = spec.strike
    T = spec.maturity
    r = spec.rate
    sigma = spec.volatility

    if T == 0:
        if spec.option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    discount = math.exp(-r * T)
    forward = S * math.exp(r * T)
    stdev = sigma * math.sqrt(T)

    if stdev == 0:
        call = discount * max(forward - K, 0.0)
    else:
        d = (forward - K) / stdev
        call = discount * ((forward - K) * normal_cdf(d) + stdev * normal_pdf(d))

    if spec.option_type == "call":
        return call
    return call - discount * (forward - K)


def simulate_terminal_prices_gbm(
    spot: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_paths: int,
    seed: int | None = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    drift = (rate - 0.5 * volatility * volatility) * maturity
    diffusion = volatility * math.sqrt(maturity) * z
    return spot * np.exp(drift + diffusion)


def monte_carlo_price(
    spec: OptionSpec,
    n_paths: int = 100_000,
    seed: int | None = 42,
) -> tuple[float, float]:
    spec.validate()

    if spec.maturity == 0:
        if spec.option_type == "call":
            payoff = max(spec.spot - spec.strike, 0.0)
        else:
            payoff = max(spec.strike - spec.spot, 0.0)
        return payoff, 0.0

    terminal = simulate_terminal_prices_gbm(
        spec.spot,
        spec.maturity,
        spec.rate,
        spec.volatility,
        n_paths,
        seed,
    )

    if spec.option_type == "call":
        payoffs = np.maximum(terminal - spec.strike, 0.0)
    else:
        payoffs = np.maximum(spec.strike - terminal, 0.0)

    discounted = np.exp(-spec.rate * spec.maturity) * payoffs
    price = float(np.mean(discounted))
    stderr = float(np.std(discounted, ddof=1) / math.sqrt(n_paths))
    return price, stderr


def finite_difference_greek(
    pricer: Callable[[OptionSpec], float],
    spec: OptionSpec,
    parameter: str,
    bump: float,
) -> float:
    if bump <= 0:
        raise ValueError("bump must be positive")

    up = OptionSpec(**vars(spec))
    down = OptionSpec(**vars(spec))

    setattr(up, parameter, getattr(up, parameter) + bump)
    setattr(down, parameter, max(getattr(down, parameter) - bump, 1e-8))

    return (pricer(up) - pricer(down)) / (2.0 * bump)


def compute_greeks(spec: OptionSpec) -> dict[str, float]:
    return {
        "delta": finite_difference_greek(black_scholes_price, spec, "spot", 1e-2),
        "vega": finite_difference_greek(black_scholes_price, spec, "volatility", 1e-4),
        "rho": finite_difference_greek(black_scholes_price, spec, "rate", 1e-4),
        "theta": -finite_difference_greek(black_scholes_price, spec, "maturity", 1e-4),
    }


def implied_volatility_from_price(
    market_price: float,
    base_spec: OptionSpec,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    low = 1e-8
    high = 5.0

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        spec = OptionSpec(
            spot=base_spec.spot,
            strike=base_spec.strike,
            maturity=base_spec.maturity,
            rate=base_spec.rate,
            volatility=mid,
            option_type=base_spec.option_type,
        )
        price = black_scholes_price(spec)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


def _finalize_plot(save_path: str | None = None, show: bool = True) -> None:
    plt.tight_layout()
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=160)
    if show:
        plt.show()
    else:
        plt.close()


def plot_volatility_sensitivity(
    spec: OptionSpec,
    bs_vol_grid: np.ndarray,
    bach_vol_grid: np.ndarray,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    bs_prices = []
    for vol in bs_vol_grid:
        updated = OptionSpec(
            spot=spec.spot,
            strike=spec.strike,
            maturity=spec.maturity,
            rate=spec.rate,
            volatility=float(vol),
            option_type=spec.option_type,
        )
        bs_prices.append(black_scholes_price(updated))

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
    terminal = simulate_terminal_prices_gbm(
        spot=spec.spot,
        maturity=spec.maturity,
        rate=spec.rate,
        volatility=spec.volatility,
        n_paths=n_paths,
    )

    plt.figure(figsize=(8, 5))
    plt.hist(terminal, bins=60, alpha=0.8, edgecolor="black")
    plt.axvline(spec.strike, color="red", linestyle="--", label="Strike")
    plt.xlabel("Terminal Price")
    plt.ylabel("Frequency")
    plt.title("Simulated Terminal Price Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    _finalize_plot(save_path=save_path, show=show)


if __name__ == "__main__":
    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        option_type="call",
    )

    print("Black-Scholes:", round(black_scholes_price(spec), 4))
    print("Bachelier:", round(bachelier_price(spec), 4))

    mc_price, mc_err = monte_carlo_price(spec, n_paths=100_000, seed=7)
    print("Monte Carlo:", round(mc_price, 4), "+/-", round(mc_err, 4))

    greeks = compute_greeks(spec)
    for name, value in greeks.items():
        print(name, round(value, 4))

    implied = implied_volatility_from_price(black_scholes_price(spec), spec)
    print("Implied vol:", round(implied, 4))
