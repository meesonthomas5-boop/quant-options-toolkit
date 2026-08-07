from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from options_pricing_toolkit import (
    OptionSpec,
    bachelier_price,
    black_scholes_price,
    implied_volatility_from_price,
    monte_carlo_price,
)


def test_black_scholes_zero_maturity_call():
    spec = OptionSpec(spot=105.0, strike=100.0, maturity=0.0, rate=0.05, volatility=0.2, option_type="call")
    assert black_scholes_price(spec) == 5.0


def test_black_scholes_zero_maturity_put():
    spec = OptionSpec(spot=95.0, strike=100.0, maturity=0.0, rate=0.05, volatility=0.2, option_type="put")
    assert black_scholes_price(spec) == 5.0


def test_put_call_parity_black_scholes():
    spec_call = OptionSpec(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, option_type="call")
    spec_put = OptionSpec(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, option_type="put")
    lhs = black_scholes_price(spec_call) - black_scholes_price(spec_put)
    rhs = spec_call.spot - spec_call.strike * math.exp(-spec_call.rate * spec_call.maturity)
    assert abs(lhs - rhs) < 1e-8


def test_bachelier_zero_maturity_matches_intrinsic():
    spec = OptionSpec(spot=102.0, strike=100.0, maturity=0.0, rate=0.02, volatility=0.3, option_type="call")
    assert bachelier_price(spec) == 2.0


def test_implied_volatility_recovers_input():
    base = OptionSpec(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, option_type="call")
    price = black_scholes_price(base)
    iv = implied_volatility_from_price(price, base)
    assert abs(iv - 0.2) < 1e-6


def test_monte_carlo_price_is_close_to_black_scholes():
    spec = OptionSpec(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2, option_type="call")
    mc_price, _ = monte_carlo_price(spec, n_paths=200_000, seed=7)
    bs_price = black_scholes_price(spec)
    assert abs(mc_price - bs_price) < 0.2
