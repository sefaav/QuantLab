"""Edge-case coverage for metrics, var, stress and exposure helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from quantlab.config import ExperimentConfig
from quantlab.risk import metrics as M
from quantlab.risk.drawdown import (
    average_drawdown,
    drawdown_durations,
    max_drawdown_details,
)
from quantlab.risk.exposure import average_gross_exposure, average_net_exposure
from quantlab.risk.stress import delay_execution, remove_best_days, scale_costs
from quantlab.risk.var import historical_cvar, historical_var

EMPTY = pd.Series(dtype=float)
ONE = pd.Series([0.01])


@pytest.mark.parametrize(
    "fn",
    [
        lambda s: M.total_return(s),
        lambda s: M.cagr(s),
        lambda s: M.annualized_volatility(s),
        lambda s: M.sharpe_ratio(s),
        lambda s: M.sortino_ratio(s),
        lambda s: M.calmar_ratio(s),
        lambda s: M.hit_rate(s),
        lambda s: M.skewness(s),
        lambda s: M.kurtosis(s),
        lambda s: M.beta(s, s),
        lambda s: M.information_ratio(s, s),
        lambda s: M.tracking_error(s, s),
        lambda s: M.annualized_alpha(s, s),
    ],
)
def test_metrics_handle_empty_and_singleton(fn: Callable[[pd.Series], float]) -> None:
    # Degenerate sample sizes (too small to estimate a statistic) must yield
    # a defined finite fallback rather than raising or returning NaN/inf.
    assert np.isfinite(fn(EMPTY))
    assert np.isfinite(fn(ONE))


def test_profit_factor_edges() -> None:
    assert M.profit_factor(pd.Series([0.1, 0.2])) == float("inf")  # no losses
    assert M.profit_factor(EMPTY) == 0.0
    assert M.profit_factor(pd.Series([0.1, -0.05])) == pytest.approx(2.0)


def test_cagr_negative_equity_floor() -> None:
    eq = pd.Series([100.0, -5.0], index=pd.date_range("2020-01-01", periods=2))
    assert M.cagr(eq) == -1.0


def test_var_cvar_empty() -> None:
    assert historical_var(EMPTY) == 0.0
    assert historical_cvar(EMPTY) == 0.0


def test_drawdown_helpers() -> None:
    # Drawdown path is [0, -0.10, -0.05, -0.20, 0.0]: one 3-period underwater
    # stretch (indices 1-3) bottoming out at -0.20 from the peak at index 0.
    eq = pd.Series([100.0, 90.0, 95.0, 80.0, 120.0])
    assert average_drawdown(eq) == pytest.approx(-0.07)
    assert drawdown_durations(eq) == [3]
    details = max_drawdown_details(eq)
    assert float(details["max_drawdown"]) == pytest.approx(-0.20)  # type: ignore[arg-type]
    assert details["peak_date"] == eq.index[0]
    assert details["trough_date"] == eq.index[3]


def test_exposure_helpers() -> None:
    # Row 1: |0.5| + |-0.2| = 0.7 gross, 0.5 + -0.2 = 0.3 net.
    # Row 2: |0.5| + |0.0| = 0.5 gross, 0.5 + 0.0 = 0.5 net.
    weights = pd.DataFrame({"A": [0.5, 0.5], "B": [-0.2, 0.0]})
    assert average_gross_exposure(weights) == pytest.approx(0.6)
    assert average_net_exposure(weights) == pytest.approx(0.4)


def test_stress_helpers(sample_config: ExperimentConfig) -> None:
    scaled = scale_costs(sample_config, commission_mult=2.0, slippage_mult=3.0)
    assert scaled.execution.commission_bps == sample_config.execution.commission_bps * 2
    assert scaled.execution.slippage_bps == sample_config.execution.slippage_bps * 3
    rets = pd.Series([0.05, -0.01, 0.03, 0.20, -0.02])
    assert remove_best_days(rets, 1).max() < rets.max()
    delayed = delay_execution(rets, 1)
    assert delayed.iloc[0] == 0.0
