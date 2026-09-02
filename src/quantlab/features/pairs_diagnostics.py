"""Shared diagnostics for a two-asset relationship (pairs trading).

The single source of truth for "is this pair a good candidate, and does its
relationship still hold" -- reused identically by the Strategy Explorer's
Pairs Trading lab, the dashboard Results tab, and the generated HTML report,
so all three agree on the same pair's hedge ratio, spread and stationarity
whenever the data range, symbols, price type and parameters they're each
given are identical (the lab lets a user explore different ones by design,
so its numbers can legitimately differ from a specific backtest's).
This extends to the PERIODIC stationarity check
(``PairDiagnostics.rolling_adf_pvalue``): it calls
``quantlab.strategies.pairs_trading.periodic_stationarity_pvalues``, the
exact same FUNCTION the live strategy's own entry gate uses, rather than a
separately-computed approximation of it -- but calling the same function is
not the same as reproducing the same result: under a mixed-calendar
universe the live gate feeds it each leg sliced to the INTERSECTION of both
legs' own native session dates, while this module (see
:func:`compute_pair_diagnostics`'s own docstring) feeds it the full
combined, closure-padded timeline. The two match exactly for a
single-calendar pair (the intersection IS the combined timeline there), and
can genuinely diverge for a mixed-calendar one -- disclosed in
docs/limitations.md, never silently assumed equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantlab.features._validation import (
    choice,
    numeric_pandas,
    positive_int,
    same_axes,
)
from quantlab.features.mean_reversion import half_life
from quantlab.features.returns import simple_returns
from quantlab.features.stationarity import (
    ADFResult,
    CointegrationResult,
    adf_test,
    cointegration_test,
)
from quantlab.features.volatility import rolling_correlation


def spread(
    a: pd.Series, b: pd.Series, intercept: pd.Series, beta: pd.Series
) -> pd.Series:
    """Residual of the trailing-OLS relationship ``a = intercept + beta * b``.

    The strategy and every diagnostic consumer share this one formula.
    """
    validated_a = numeric_pandas(a, name="a")
    validated_b = numeric_pandas(b, name="b")
    validated_intercept = numeric_pandas(intercept, name="intercept")
    validated_beta = numeric_pandas(beta, name="beta")
    same_axes(
        validated_a,
        validated_b,
        validated_intercept,
        validated_beta,
        names=("b", "intercept", "beta"),
    )
    return validated_a - validated_intercept - validated_beta * validated_b


@dataclass(frozen=True)
class PairDiagnostics:
    """A snapshot AND a time-history of a two-asset relationship's quality.

    ``adf_result``/``cointegration_result``/``half_life`` are EXPLORATORY:
    a single ADF/Engle-Granger test over the whole sample's spread/series,
    useful for an initial read on the relationship but not what the live
    strategy actually gates trading decisions on. ``rolling_adf_pvalue`` is
    the CAUSAL one: it reproduces, bar for bar,
    :meth:`~quantlab.strategies.pairs_trading.PairsTradingStrategy.
    _stationarity_gate`'s own periodic recheck (a fresh single-window
    regression fit every ``rolling_adf_stride`` periods, exactly like the
    strategy) via the shared
    :func:`~quantlab.strategies.pairs_trading.periodic_stationarity_pvalues`
    -- so this field can never show a different picture than what actually
    gated (or would have gated) new entries in a real backtest of this
    pair. ``hedge_ratio_stability`` likewise describes stability over time
    rather than a one-off snapshot.
    """

    symbol_a: str
    symbol_b: str
    correlation: float
    rolling_correlation: pd.Series
    hedge_ratio: pd.Series
    intercept: pd.Series
    spread: pd.Series
    # Which indicator `spread_indicator` was actually computed with --
    # zscore, rsi or percentile (see
    # `quantlab.strategies.pairs_trading.INDICATORS`), NOT always zscore.
    # A consumer must label charts/tables from this field rather than
    # hardcoding "Z-score", or the display would misrepresent an
    # rsi/percentile-configured pair as a zscore-driven one.
    indicator: str
    spread_indicator: pd.Series
    adf_result: ADFResult | None
    cointegration_result: CointegrationResult | None
    half_life: float
    hedge_ratio_stability: float
    rolling_adf_pvalue: pd.Series


def compute_pair_diagnostics(
    prices: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    *,
    formation_window: int,
    indicator_window: int,
    dynamic_hedge_ratio: bool,
    indicator: str = "zscore",
    correlation_window: int | None = None,
    rolling_adf_stride: int | None = None,
) -> PairDiagnostics:
    """Compute the full diagnostic picture for one candidate pair.

    ``indicator`` selects the SAME zscore/rsi/percentile series
    :class:`~quantlab.strategies.pairs_trading.PairsTradingStrategy` itself
    can be configured with (via
    :func:`~quantlab.strategies.pairs_trading._centered_spread_indicator`,
    the exact function the live strategy uses) -- defaults to ``"zscore"``
    only for a caller that has no strategy instance to read a configured
    indicator from (e.g. exploring a candidate pair before choosing one).
    ``correlation_window``/``rolling_adf_stride`` default to
    ``indicator_window`` -- the same cadence the strategy itself already
    uses for its own periodic stationarity gate.

    Under a mixed-calendar universe, ``hedge_ratio``/``spread``/
    ``spread_indicator``/``rolling_adf_pvalue`` here are ALL computed on
    the FULL combined, closure-padded timeline, NOT the intersection of
    both legs' own native session dates
    :meth:`~quantlab.strategies.pairs_trading.PairsTradingStrategy.
    _native_pair_context` uses for the live strategy -- disclosed in
    docs/limitations.md rather than silently assumed equivalent.
    ``rolling_adf_pvalue`` calls the exact same gated
    ``periodic_stationarity_pvalues`` FUNCTION the live entry gate uses
    (see :class:`PairDiagnostics`'s own docstring), but that alone does
    not make its RESULT match: the live gate feeds that function the
    native-intersection-sliced series, this function feeds it the
    combined-timeline ``a``/``b`` above -- the two match exactly for a
    single-calendar pair and can genuinely diverge for a mixed-calendar
    one, same as every other diagnostic here.
    """
    from quantlab.strategies.pairs_trading import (
        INDICATORS,
        _centered_spread_indicator,
        periodic_stationarity_pvalues,
        rolling_hedge_parameters,
    )

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    for symbol in (symbol_a, symbol_b):
        if symbol not in prices.columns:
            raise ValueError(f"prices is missing symbol {symbol!r}.")
    validated_indicator = choice(
        indicator, name="indicator", options=frozenset(INDICATORS)
    )
    a = numeric_pandas(
        prices[symbol_a], name="prices[symbol_a]", strictly_positive=True
    )
    b = numeric_pandas(
        prices[symbol_b], name="prices[symbol_b]", strictly_positive=True
    )
    same_axes(a, b, names=("prices[symbol_b]",))
    corr_window = positive_int(
        correlation_window if correlation_window is not None else indicator_window,
        name="correlation_window",
        minimum=2,
    )
    stride = positive_int(
        rolling_adf_stride if rolling_adf_stride is not None else indicator_window,
        name="rolling_adf_stride",
        minimum=1,
    )

    returns_a = simple_returns(a)
    returns_b = simple_returns(b)
    correlation = float(returns_a.corr(returns_b))
    rolling_corr = rolling_correlation(returns_a, returns_b, corr_window)

    intercept, beta = rolling_hedge_parameters(
        a, b, formation_window, dynamic_hedge_ratio
    )
    spread_series = spread(a, b, intercept, beta)
    indicator_series = _centered_spread_indicator(
        spread_series, validated_indicator, indicator_window
    )

    clean_spread = spread_series.dropna()
    adf_result = adf_test(clean_spread)
    cointegration_result = cointegration_test(a, b)
    half_life_estimate = half_life(clean_spread)

    clean_beta = beta.dropna()
    hedge_ratio_stability = (
        float(clean_beta.std(ddof=1)) if len(clean_beta) > 1 else float("nan")
    )

    rolling_pvalue = periodic_stationarity_pvalues(
        a,
        b,
        formation_window=formation_window,
        stride=stride,
        dynamic_hedge_ratio=dynamic_hedge_ratio,
    )

    return PairDiagnostics(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        correlation=correlation,
        rolling_correlation=rolling_corr,
        hedge_ratio=beta,
        intercept=intercept,
        spread=spread_series,
        indicator=validated_indicator,
        spread_indicator=indicator_series,
        adf_result=adf_result,
        cointegration_result=cointegration_result,
        half_life=half_life_estimate,
        hedge_ratio_stability=hedge_ratio_stability,
        rolling_adf_pvalue=rolling_pvalue,
    )
