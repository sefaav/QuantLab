"""Shared pytest fixtures.

The synthetic-data generators here are deterministic (seeded) so every test is
reproducible. They produce data in the canonical long OHLCV format
that the rest of the pipeline expects.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.config import ExperimentConfig
from quantlab.constants import (
    ADJUSTED_CLOSE,
    CLOSE,
    HIGH,
    LOW,
    OPEN,
    SYMBOL,
    TIMESTAMP,
    VOLUME,
)


def make_ohlcv(
    symbol: str,
    prices: np.ndarray | Sequence[float],
    start: str = "2020-01-01",
    freq: str = "B",
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """Build a canonical long OHLCV frame for one symbol from a close series.

    open/high/low are derived from close with a small, deterministic spread so
    the frame satisfies the validation invariants (high >= max(o,c), etc.).
    """
    n = len(prices)
    idx = pd.date_range(start=start, periods=n, freq=freq)
    close = np.asarray(prices, dtype=float)
    # Deterministic, invariant-safe OHLC around the close.
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]  # open = previous close
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    return pd.DataFrame(
        {
            TIMESTAMP: idx,
            SYMBOL: symbol,
            OPEN: open_,
            HIGH: high,
            LOW: low,
            CLOSE: close,
            ADJUSTED_CLOSE: close,  # synthetic: no dividends/splits
            VOLUME: float(volume),
        }
    )


def geometric_series(
    n: int, mu: float, sigma: float, s0: float, seed: int
) -> np.ndarray:
    """Deterministic geometric random-walk close prices."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu, sigma, size=n)
    return float(s0) * np.exp(np.cumsum(shocks))


@pytest.fixture
def rising_prices() -> np.ndarray:
    """Strictly increasing prices — momentum signal must be positive."""
    return np.linspace(100.0, 200.0, 120)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """A 3-symbol canonical long OHLCV panel with distinct dynamics."""
    n = 400
    a = geometric_series(n, mu=0.0008, sigma=0.010, s0=100.0, seed=1)  # trend up
    b = geometric_series(n, mu=-0.0003, sigma=0.012, s0=100.0, seed=2)  # down
    c = geometric_series(n, mu=0.0002, sigma=0.020, s0=100.0, seed=3)  # volatile
    frames = [
        make_ohlcv("AAA", a),
        make_ohlcv("BBB", b),
        make_ohlcv("CCC", c),
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def two_symbol_panel() -> pd.DataFrame:
    """Two cointegrated-ish symbols for pairs-trading tests."""
    n = 500
    common = geometric_series(n, mu=0.0004, sigma=0.010, s0=100.0, seed=7)
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.5, size=n)
    a = common
    b = 0.8 * common + 20.0 + noise  # b tracks a with noise around a spread
    return pd.concat([make_ohlcv("EWA", a), make_ohlcv("EWB", b)], ignore_index=True)


@pytest.fixture
def sample_config() -> ExperimentConfig:
    """A small, valid config usable without any network access."""
    return ExperimentConfig.from_dict(
        {
            "experiment_name": "unit_test_experiment",
            "data": {
                "source": "csv",
                "symbols": ["AAA", "BBB", "CCC"],
                "start_date": date(2020, 1, 1),
                "end_date": date(2021, 8, 1),
                "frequency": "1d",
                "missing_value_policy": "drop",
                "market_calendar": "XNYS",
            },
            "strategy": {"name": "buy_and_hold", "parameters": {}},
            "portfolio": {
                "allocator": "equal_weight",
                "maximum_weight": 1.0,
                "rebalance_frequency": "monthly",
            },
            "execution": {
                "commission_bps": 2.0,
                "spread_bps": 3.0,
                "slippage_bps": 2.0,
            },
            "backtest": {
                "initial_capital": 100_000.0,
                "benchmark_symbol": "AAA",
                "risk_free_rate": 0.0,
                "periods_per_year": 252,
            },
            "reproducibility": {"random_seed": 42},
        }
    )
