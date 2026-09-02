"""Cell definitions consumed by ``scripts/build_notebooks.py``."""

from __future__ import annotations

NB_01_DATA_QUALITY = [
    (
        "md",
        """\
# 01 — Data Quality

Loads the example experiment's market data through ``DataLoader``, which uses
the local cache and downloads missing ranges when the source is available.
It then runs the ``DataValidator`` and inspects coverage and basic statistics
before any research begins.

This step exists because a backtest is only as trustworthy as its input data: duplicates, gaps, and impossible OHLC bars must be
caught before they silently distort results.
""",
    ),
    (
        "code",
        """\
%matplotlib inline
import matplotlib.pyplot as plt

from quantlab.config import ExperimentConfig
from quantlab.data.loader import DataLoader
from quantlab.data.base import price_matrix

config = ExperimentConfig.from_yaml("../configs/momentum_sp500.yaml")
config.experiment_name, config.symbols, config.start_date, config.end_date
""",
    ),
    (
        "code",
        """\
data, report = DataLoader().load(config)
print(report.summary())
for w in report.warnings:
    print(" -", w)
""",
    ),
    ("md", "## Coverage per symbol"),
    (
        "code",
        """\
data.groupby("symbol")["timestamp"].agg(["min", "max", "count"])
""",
    ),
    ("md", "## Adjusted close price series"),
    (
        "code",
        """\
prices = price_matrix(data)
fig, ax = plt.subplots(figsize=(10, 5))
(prices / prices.iloc[0] * 100).plot(ax=ax, lw=1.2)
ax.set_title("Rebased adjusted close (100 = start of sample)")
ax.set_ylabel("Index level")
ax.legend(ncol=4, fontsize=8)
plt.show()
""",
    ),
    ("md", "## Daily return summary statistics"),
    (
        "code",
        """\
returns = prices.pct_change(fill_method=None)
returns.describe().T[["mean", "std", "min", "max"]].style.format("{:.4f}")
""",
    ),
    (
        "md",
        """\
## Takeaways

- No duplicate `(timestamp, symbol)` rows and no non-positive prices were
  found (or they would appear as warnings above).
- Coverage spans the full requested range for every symbol.
- Returns look like ordinary daily equity/ETF returns (no obviously broken
  bars). This is the baseline the rest of the research builds on.
""",
    ),
]


NB_02_MOMENTUM_RESEARCH = [
    (
        "md",
        """\
# 02 — Cross-Sectional Momentum Research (Example)

**Research question**: Can cross-sectional momentum generate stable
out-of-sample risk-adjusted returns across liquid multi-asset ETFs after
transaction costs and volatility targeting?

**Universe**: SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ — US large/small
cap, developed/emerging ex-US equities, long treasuries, gold, real estate.

**Signal**: 12-month return excluding the most recent month (standard
momentum construction to avoid short-term reversal), ranked cross-sectionally
each month; long the top half.

**Hypothesis** (H1): after realistic transaction costs and out-of-sample
validation, the strategy delivers a positive risk-adjusted return (Sharpe > 0)
that is reasonably stable across sub-periods.
""",
    ),
    (
        "code",
        """\
%matplotlib inline
import matplotlib.pyplot as plt

from quantlab.config import ExperimentConfig
from quantlab.data.loader import DataLoader
from quantlab.backtesting.runner import run_backtest_from_config

config = ExperimentConfig.from_yaml("../configs/momentum_sp500.yaml")
data, report = DataLoader().load(config)
len(data), report.row_count
""",
    ),
    ("md", "## Run the backtest"),
    (
        "code",
        """\
result = run_backtest_from_config(data, config, data_quality_report=report)
print(result.summary())
""",
    ),
    ("md", "## Equity curve vs benchmark"),
    (
        "code",
        """\
equity = result.equity_curve
benchmark_returns = result.benchmark_returns

if benchmark_returns is None:
    raise RuntimeError("No benchmark returns are available for this backtest.")

bench_equity = float(equity.iloc[0]) * (1 + benchmark_returns).cumprod()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(equity.index, equity.to_numpy(dtype=float), label="Strategy", lw=1.5)
ax.plot(bench_equity.index, bench_equity.to_numpy(dtype=float), label=f"Benchmark ({config.benchmark_label})", lw=1.2, ls="--")
ax.set_title("Cross-sectional momentum vs benchmark")
ax.legend()
plt.show()
""",
    ),
    ("md", "## Performance by year"),
    (
        "code",
        """\
from quantlab.reporting.tables import subperiod_table
table = subperiod_table(result)
table
""",
    ),
    (
        "md",
        """\
## Out-of-sample validation (walk-forward)

Parameters are selected on a validation block and only then evaluated on an
untouched test block — the test period is never used to choose
`lookback_period`, `skip_period` or `top_fraction`. The notebook calls the same
configured-or-default grid helper as the CLI, so both interfaces evaluate
identical candidates.
""",
    ),
    (
        "code",
        """\
from quantlab.validation import WalkForwardValidator, parameter_grid_for_config

train_window = config.validation.train_window or 500
validation_window = config.validation.validation_window or 126
test_window = config.validation.test_window or 126

validator = WalkForwardValidator(config)
wf = validator.run(
    data,
    parameter_grid=parameter_grid_for_config(config),
    train_window=train_window,
    validation_window=validation_window,
    test_window=test_window,
    expanding=config.validation.expanding,
)
print(f"{len(wf.folds)} walk-forward folds")
wf.summary_table()
""",
    ),
    (
        "code",
        """\
oos_metrics = wf.oos_metrics(config.periods_per_year, config.risk_free_rate)
print(f"Out-of-sample Sharpe: {oos_metrics.get('sharpe_ratio', 0):.2f}")
print(f"Out-of-sample CAGR:   {oos_metrics.get('cagr', 0):.2%}")
print(f"Parameter stability (coefficient of variation): {wf.parameter_stability()}")
""",
    ),
    (
        "code",
        """\
%matplotlib inline

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(wf.oos_equity.index, wf.oos_equity.to_numpy(dtype=float), color="#2563eb", lw=1.3)
ax.set_title("Stitched out-of-sample equity curve (walk-forward test blocks only)")
plt.show()
""",
    ),
    (
        "md",
        """\
## Conclusion

The strategy's full-sample and walk-forward out-of-sample Sharpe ratios are
relatively close, which provides some reassurance against severe overfitting.
However, the observed risk-adjusted performance remains modest and does not
demonstrate a persistent ability to outperform the market. These results are
historical, depend on the stated assumptions and execution costs, and do not
guarantee future performance.
""",
    ),
]


NB_03_MEAN_REVERSION_RESEARCH = [
    (
        "md",
        """\
# 03 — Mean Reversion Research (Example)

Compares three mean-reversion indicators — RSI, Bollinger %b, and a rolling
z-score — on the same ETF universe, then evaluates the z-score strategy
end-to-end through the backtest engine.
""",
    ),
    (
        "code",
        """\
%matplotlib inline
import matplotlib.pyplot as plt

from quantlab.config import ExperimentConfig
from quantlab.data.loader import DataLoader
from quantlab.data.base import price_matrix
from quantlab.features.mean_reversion import rolling_zscore, rsi, bollinger_bands
from quantlab.backtesting.runner import run_backtest_from_config

config = ExperimentConfig.from_yaml("../configs/mean_reversion_etfs.yaml")
data, report = DataLoader().load(config)
prices = price_matrix(data)
spy = prices["SPY"].dropna()
print(report.summary())
""",
    ),
    ("md", "## Comparing indicators on SPY"),
    (
        "code",
        """\
z = rolling_zscore(spy, window=20)
r = rsi(spy, window=14)
bb = bollinger_bands(spy, window=20, num_std=2.0)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(z.index, z.to_numpy(dtype=float), color="#2563eb", lw=0.9)
axes[0].axhline(2, color="#dc2626", lw=0.8, ls="--")
axes[0].axhline(-2, color="#16a34a", lw=0.8, ls="--")
axes[0].set_title("Rolling z-score (20d)")

axes[1].plot(r.index, r.to_numpy(dtype=float), color="#f59e0b", lw=0.9)
axes[1].axhline(70, color="#dc2626", lw=0.8, ls="--")
axes[1].axhline(30, color="#16a34a", lw=0.8, ls="--")
axes[1].set_title("RSI (14d)")

axes[2].plot(bb.index, bb["pct_b"].to_numpy(dtype=float), color="#6b7280", lw=0.9)
axes[2].axhline(1.0, color="#dc2626", lw=0.8, ls="--")
axes[2].axhline(0.0, color="#16a34a", lw=0.8, ls="--")
axes[2].set_title("Bollinger %b (20d, 2 std)")
plt.tight_layout()
plt.show()
""",
    ),
    (
        "md",
        """\
The three indicators broadly agree on *when* SPY is stretched (their extremes
line up in time), but disagree on magnitude — this is exactly why the
strategy config exposes `entry_threshold` / `exit_threshold` as tunable
parameters (on whichever `indicator` is selected — z-score, Bollinger %B, RSI,
distance to a moving average, or percentile rank) rather than hard-coding
one indicator's convention.
""",
    ),
    ("md", "## Full backtest: rolling z-score mean reversion"),
    (
        "code",
        """\
result = run_backtest_from_config(data, config, data_quality_report=report)
print(result.summary())
""",
    ),
    (
        "code",
        """\
from quantlab.risk.drawdown import drawdown_series
dd = drawdown_series(result.equity_curve)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
axes[0].plot(result.equity_curve.index, result.equity_curve.to_numpy(dtype=float), color="#2563eb", lw=1.3)
axes[0].set_title("Mean-reversion equity curve")
axes[1].fill_between(dd.index, dd.to_numpy(dtype=float), 0, color="#dc2626", alpha=0.3)
axes[1].set_title("Drawdown")
plt.tight_layout()
plt.show()
""",
    ),
    ("md", "## Gross vs net performance"),
    (
        "code",
        """\
from quantlab.reporting.tables import gross_net_table

gross_net_table(result)
""",
    ),
    (
        "md",
        """\
## Takeaways

Short-horizon mean reversion on liquid equity ETFs trades far more often than
the monthly-rebalanced momentum strategy (see the trade count above), so
transaction costs matter proportionally more here. The gross-versus-net table
above quantifies how much of the strategy's pre-cost return is lost to
transaction costs.
""",
    ),
]


NB_04_PAIRS_TRADING_RESEARCH = [
    (
        "md",
        """\
# 04 — Pairs Trading Research (Example)

**Pair**: EWA (Australia) / EWC (Canada) — two commodity-linked developed
markets, chosen for economic reasoning rather than cherry-picked for
historical performance.
""",
    ),
    (
        "code",
        """\
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from quantlab.config import ExperimentConfig
from quantlab.data.loader import DataLoader
from quantlab.data.base import price_matrix
from quantlab.strategies.pairs_trading import adf_pvalue, rolling_hedge_parameters
from quantlab.features.mean_reversion import rolling_zscore
from quantlab.backtesting.runner import run_backtest_from_config

config = ExperimentConfig.from_yaml("../configs/pairs_trading.yaml")
data, report = DataLoader().load(config)
prices = price_matrix(data)
a, b = prices["EWA"].dropna(), prices["EWC"].dropna()
common_idx = a.index.intersection(b.index)
a, b = a.loc[common_idx], b.loc[common_idx]
print(report.summary())
print(f"Correlation of daily returns: {a.pct_change().corr(b.pct_change()):.3f}")
""",
    ),
    ("md", "## Hedge ratio and spread"),
    (
        "code",
        """\
intercept, beta = rolling_hedge_parameters(a, b, window=252, dynamic=True)
spread = a - intercept - beta * b

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(beta.index, beta.to_numpy(dtype=float), color="#2563eb", lw=1.0)
axes[0].set_title("Rolling hedge ratio (beta), 252d window")
axes[1].plot(spread.index, spread.to_numpy(dtype=float), color="#6b7280", lw=1.0)
axes[1].set_title("Regression residual = EWA - intercept - beta * EWC")
plt.tight_layout()
plt.show()
""",
    ),
    (
        "md",
        """\
## Stationarity of the spread (Augmented Dickey-Fuller)

The Augmented Dickey-Fuller test evaluates whether the spread behaves like a
non-stationary unit-root process or tends to remain around a more stable
long-term level.

- **Null hypothesis (H0):** the spread has a unit root and is non-stationary;
  deviations may persist indefinitely.
- **Alternative hypothesis (H1):** the spread is stationary; deviations are
  more consistent with mean-reverting behaviour.

A low p-value provides evidence against the unit-root hypothesis. For example,
a p-value below 0.05 allows H0 to be rejected at the conventional 5% level.
It does not represent the probability that the spread is stationary and does
not prove that a trading strategy will be profitable.

The test below uses all available regression residuals and is therefore
descriptive rather than a tradable signal. During the backtest, QuantLab
periodically recomputes the regression and ADF test on exactly the preceding
formation window. The result acts as an entry gate: a new position is
permitted only when the residual passes the configured stationarity threshold
and the z-score entry condition is also satisfied.
""",
    ),
    (
        "code",
        """\
pvalue_full_sample = adf_pvalue(spread.dropna())
if pvalue_full_sample is None:
    print("ADF test inconclusive on the full-sample regression residual.")
else:
    print(f"ADF p-value on the full-sample residual: {pvalue_full_sample:.2e}")
    print("(A low value is evidence against the unit-root null.)")
""",
    ),
    (
        "code",
        """\
# The rolling regression above re-estimates the hedge ratio every day, which
# absorbs any drift in the true EWA/EWC relationship into a time-varying beta
# instead of leaving it in the residual — this alone can make the residual
# look far more stationary than it really is. Comparing against a *single*
# hedge ratio fixed over the whole sample shows how much of the result above
# comes from that adaptivity rather than genuine long-run cointegration.
from quantlab.strategies.pairs_trading import _ols_coefficients

static_intercept, static_beta = _ols_coefficients(
    b.to_numpy(dtype=float), a.to_numpy(dtype=float)
)
static_spread = a - static_intercept - static_beta * b
pvalue_static = adf_pvalue(static_spread.dropna())

print(f"Rolling-hedge-ratio spread ADF p-value: {pvalue_full_sample:.2e}")
if pvalue_static is None:
    print("Static (single, full-sample) hedge-ratio spread: ADF test inconclusive.")
else:
    print(f"Static (single, full-sample) hedge-ratio spread ADF p-value: {pvalue_static:.4f}")
""",
    ),
    (
        "md",
        """\
The two p-values above can differ by orders of magnitude, and that is
expected, not a bug. The rolling regression re-fits its hedge ratio to the
*current* relationship every day, so any long-run drift between EWA and EWC
is absorbed into the time-varying beta rather than left in the residual —
which makes a rolling-hedge spread mechanically look more stationary than a
spread built from one hedge ratio fixed over the whole sample. Neither number
is "the" answer to "is this pair cointegrated": the rolling version matches
what `dynamic_hedge_ratio: true` does during the backtest (see
`configs/pairs_trading.yaml`), while the static version is the more
conservative, classical full-sample cointegration test. Read a very small
rolling-spread p-value as "this hedge ratio tracks the relationship well,"
not as strong standalone evidence that the pair is permanently stationary.
""",
    ),
    ("md", "## Spread z-score and trading thresholds"),
    (
        "code",
        """\
z = rolling_zscore(spread, window=config.strategy.parameters["indicator_window"])
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(z.index, z.to_numpy(dtype=float), color="#2563eb", lw=0.8)
ax.axhline(2, color="#dc2626", lw=0.8, ls="--")
ax.axhline(-2, color="#16a34a", lw=0.8, ls="--")
ax.axhline(0, color="#9ca3af", lw=0.6)
ax.set_title("Pairs spread z-score with entry thresholds (+/-2)")
plt.show()
""",
    ),
    ("md", "## Full backtest"),
    (
        "code",
        """\
result = run_backtest_from_config(data, config, data_quality_report=report)
print(result.summary())
""",
    ),
    (
        "md",
        """\
## Takeaways

The spread exhibits periods of mean-reverting behaviour, but the strategy
produces a negative net Sharpe ratio after transaction costs. Under the tested
rules, period and execution assumptions, the results therefore provide no
evidence of a profitable trading edge. This remains an honest and useful
research finding: statistical mean reversion does not necessarily translate
into an economically viable strategy after costs.
""",
    ),
]


NB_05_ROBUSTNESS_ANALYSIS = [
    (
        "md",
        """\
# 05 — Robustness Analysis (Example)

Parameter sensitivity, bootstrap resampling, stress tests and a Monte Carlo
permutation check for the example cross-sectional momentum strategy. The goal is to find **stable regions** of parameter space and to gauge
whether the result could plausibly be noise — not to find one lucky optimum.
""",
    ),
    (
        "code",
        """\
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from quantlab.config import ExperimentConfig
from quantlab.data.loader import DataLoader
from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.validation.parameter_sensitivity import (
    run_parameter_sensitivity, sensitivity_heatmap_data,
)
from quantlab.validation.bootstrap import bootstrap_returns
from quantlab.validation.robustness import run_stress_tests, monte_carlo_permutation
from quantlab.reporting.tables import format_bootstrap_summary

config = ExperimentConfig.from_yaml("../configs/momentum_sp500.yaml")
data, report = DataLoader().load(config)
result = run_backtest_from_config(data, config, data_quality_report=report)
result.metrics["sharpe_ratio"], result.metrics["cagr"]
""",
    ),
    ("md", "## Parameter sensitivity: lookback period x top fraction"),
    (
        "code",
        """\
sensitivity = run_parameter_sensitivity(
    data, config,
    parameter_x="lookback_period", values_x=[126, 189, 252, 315],
    parameter_y="top_fraction", values_y=[0.25, 0.375, 0.5, 0.625],
)
heat = sensitivity_heatmap_data(sensitivity, "lookback_period", "top_fraction", "sharpe")
heat
""",
    ),
    (
        "code",
        """\
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(heat.to_numpy(dtype=float), cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns)
ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index)
ax.set_xlabel("lookback_period"); ax.set_ylabel("top_fraction")
ax.set_title("Sharpe ratio across parameter combinations")
fig.colorbar(im, ax=ax, label="Sharpe")
plt.show()
""",
    ),
    (
        "md",
        """\
The goal here is a broad plateau of reasonable Sharpe ratios, not a single
sharp peak — a plateau suggests the result isn't a fragile accident of one
specific parameter choice.
""",
    ),
    (
        "md",
        """\
## Bootstrap distribution of CAGR, Sharpe, maximum drawdown

This test asks a simple question: would the strategy's results remain similar
if the same types of gains and losses had occurred in a different sequence?

QuantLab starts from the strategy's historical net daily returns. For each of
1,000 simulations, it selects a random starting date and copies 21 consecutive
returns, roughly one trading month. It repeats this process until it has built
a synthetic history with the same length as the original one.

A historical block may appear several times in one simulation or may not be
selected at all. Sampling consecutive blocks, rather than isolated days,
preserves some short periods of high volatility, low volatility, gains and
losses.

QuantLab then calculates the CAGR, Sharpe ratio, maximum drawdown and final
portfolio value for every synthetic history.
""",
    ),
    (
        "code",
        """\
boot = bootstrap_returns(
    result.returns, n_iterations=1000, block_size=21, seed=config.random_seed,
    periods_per_year=config.periods_per_year, initial_capital=config.initial_capital,
    risk_free_rate=config.risk_free_rate,
)
format_bootstrap_summary(boot.summary())
""",
    ),
    (
        "md",
        """\
The median represents the typical result across the 1,000 synthetic histories.
The `p_lower` and `p_upper` columns contain the middle 90% of the simulated
outcomes (the 5th/95th percentiles by default; configurable via
`robustness.bootstrap.confidence_level`):

- for CAGR, Sharpe and final value, `p_lower` is the less favourable boundary
  and `p_upper` is the more favourable boundary;
- for maximum drawdown, a more negative value represents a worse loss, so
  `p_lower` is the more severe drawdown scenario.

A narrow range suggests that the result is relatively insensitive to the
historical ordering of returns. A wide range means that performance depends
strongly on when gains and losses occur.

This is not a forecast of future performance. The bootstrap rearranges returns
already observed in the backtest; it does not create new market crises, rerun
the trading strategy or model a permanent change in market behaviour.
""",
    ),
    ("md", "## Stress tests"),
    (
        "code",
        """\
stress = run_stress_tests(data, config)
stress
""",
    ),
    (
        "md",
        """\
## Monte Carlo permutation test

This Monte Carlo sign-flip test asks whether the observed Sharpe ratio is
unusually high relative to a simple zero-direction null model. Each simulation
keeps the magnitude and date of every realised strategy return but randomly
assigns it a positive or negative sign. This destroys the strategy's historical
directional structure while preserving the sequence of return magnitudes.

The empirical p-value is the fraction of randomised simulations whose Sharpe
ratio is at least as high as the real one. A low value indicates that the
observed Sharpe is difficult to reproduce under this specific random-sign
hypothesis. This suggests that the result contains more structure than would
typically be produced by randomly assigning the signs of the observed returns.
However, it is not proof that the strategy is genuine, profitable or likely
to work in the future.

The test deliberately uses a zero risk-free rate because the returns are
sign-flipped around zero. Its reported real Sharpe may therefore differ from
the risk-free-adjusted Sharpe shown elsewhere in the project.
""",
    ),
    (
        "code",
        """\
mc = monte_carlo_permutation(result.returns, n_iterations=1000, seed=config.random_seed,
                              periods_per_year=config.periods_per_year)
print(f"Real Sharpe: {mc['real_sharpe']:.3f}")
print(f"Empirical p-value (fraction of random sign-flips scoring >= real): {mc['p_value']:.3f}")
""",
    ),
    (
        "md",
        """\
## Interpretation

- A low permutation p-value suggests the realised Sharpe is unlikely to be
  pure luck under random sign structure — **this is a sanity check against
  randomness, not proof of a future edge**.
- The stress table shows how much the edge erodes under 2x/5x commissions,
  2x slippage, an extra period of execution delay, and with the best 10 days
  removed. Any strategy whose performance depends entirely on a handful of
  lucky days or on unrealistically low costs should be treated with
  suspicion.
- Combined with the walk-forward results in notebook 02, this analysis
  supports treating the strategy's historical edge as modest but not
  obviously an artifact of one lucky parameter choice — while still being, as
  always, no guarantee of future performance.
""",
    ),
]
