# Quant Options Toolkit

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/dependency-numpy-informational)
![Matplotlib](https://img.shields.io/badge/dependency-matplotlib-informational)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A compact Python project for pricing European options, computing simple Greeks, estimating implied volatility, and comparing closed-form models with Monte Carlo simulation.

## What it does

- Black-Scholes pricing for European calls and puts
- Bachelier pricing for European calls and puts
- Monte Carlo pricing under geometric Brownian motion
- Finite-difference Greeks
- Implied volatility by bisection
- Volatility sensitivity plots
- Terminal price distribution plots

## Why this project exists

This is meant to be a clean, interview-friendly quant project: small enough to read quickly, but broad enough to show modelling, simulation, numerical methods, and clear implementation.

## Repository layout

```text
quant-options-toolkit/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   └── options_pricing_toolkit.py
├── examples/
│   └── demo.py
└── .gitignore
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the example

```bash
python examples/demo.py
```

This prints a small summary to the console and opens two plots.

## Example output

```text
Black-Scholes call price: 10.4506
Bachelier call price: 7.9788
Monte Carlo call price: 10.4731 ± 0.0463
Greeks:
  delta: 0.6368
  vega: 37.5240
  rho: 53.2310
  theta: -6.4140
Implied volatility from BS price: 0.2000
```

## Example plots

Add screenshots here after running the demo, for example:

- `docs/volatility_sensitivity.png`
- `docs/terminal_distribution.png`

You can then embed them like this:

```markdown
![Volatility sensitivity](docs/volatility_sensitivity.png)
![Terminal distribution](docs/terminal_distribution.png)
```

## Notes

- Black-Scholes assumes lognormal dynamics.
- Bachelier assumes normal dynamics.
- The Greeks here use finite differences rather than closed-form formulas.
- Implied volatility is recovered with a simple bisection routine.

## Limitations

- European options only
- No dividends or carry adjustments
- No calibration framework
- Monte Carlo uses plain sampling, without variance reduction
- Greeks are numerical rather than analytic

## Possible extensions

- Binomial tree pricing
- Barrier and Asian options
- Variance reduction for Monte Carlo
- Delta hedging PnL simulation
- Local volatility or stochastic volatility models
- Historical backtesting of delta hedging

## What to say about it in an interview

A good short description is:

> I built a small Python toolkit for pricing European options with Black-Scholes, Bachelier, and Monte Carlo methods. I added finite-difference Greeks, implied volatility by bisection, and a couple of plots to compare models and inspect simulation output. The project was mainly a way to turn basic derivatives theory into a clean, testable piece of code.
