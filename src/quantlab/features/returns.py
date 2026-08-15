"""Return computations.

All functions accept either a ``pd.Series`` (single asset) or a ``pd.DataFrame``
(wide ``dates × symbols``) and return the same shape, preserving the index.

Look-ahead safety: ``simple_returns`` / ``log_returns`` at row *t* use only
prices up to and including *t*. ``forward_returns`` deliberately looks ahead and
is therefore **for labelling only** — it must never feed a same-date trading
decision.
"""

from __future__ import annotations

from typing import TypeVar, cast

import numpy as np
import pandas as pd

from quantlab.features._validation import finite_real, numeric_pandas, positive_int

# Series or DataFrame — return-shape preserving.
PandasT = TypeVar("PandasT", pd.Series, pd.DataFrame)


def simple_returns(prices: PandasT, periods: int = 1) -> PandasT:
    """Simple (arithmetic) returns ``P_t / P_{t-periods} - 1``.

    Example: prices ``[100, 110, 99]`` → ``[NaN, 0.10, -0.10]``.
    """
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    lag = positive_int(periods, name="periods")
    return validated.pct_change(periods=lag, fill_method=None)


def log_returns(prices: PandasT, periods: int = 1) -> PandasT:
    """Logarithmic returns ``ln(P_t / P_{t-periods})``."""
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    lag = positive_int(periods, name="periods")
    shifted = validated.shift(lag)
    # NumPy preserves the pandas container through ``__array_ufunc__``.
    return cast(PandasT, np.log(validated / shifted))


def forward_returns(prices: PandasT, periods: int = 1) -> PandasT:
    """Forward return realised *after* the current bar — **labels only**.

    ``forward_return_t = P_{t+periods} / P_t - 1``. Using this in a signal at
    date *t* is look-ahead bias; it exists solely to build supervised
    labels for research, never to trade the same bar.
    """
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    horizon = positive_int(periods, name="periods")
    return validated.shift(-horizon) / validated - 1.0


def _prepare_compounding_returns(returns: PandasT) -> PandasT:
    """Fill only leading warm-up gaps and preserve internal missing returns."""
    validated = numeric_pandas(returns, name="returns")
    if isinstance(validated, pd.Series):
        valid = validated.notna()
        seen_valid = valid.cummax()
        leading = (
            ~seen_valid if bool(valid.any()) else pd.Series(False, index=valid.index)
        )
        return cast(  # type: ignore[redundant-cast]
            PandasT, validated.mask(leading, 0.0)
        )

    valid_frame = validated.notna()
    seen_valid_frame = valid_frame.cummax()
    has_valid = valid_frame.any(axis=0)
    leading_frame = (~seen_valid_frame).mul(has_valid, axis="columns").astype(bool)
    return cast(  # type: ignore[redundant-cast]
        PandasT, validated.mask(leading_frame, 0.0)
    )


def cumulative_returns(returns: PandasT) -> PandasT:
    """Cumulative compounded growth ``prod(1 + r) - 1`` from period returns.

    Leading warm-up NaNs are treated as zero. An internal missing return makes
    the compounded path undefined from that point onward.
    """
    prepared = _prepare_compounding_returns(returns)
    return (1.0 + prepared).cumprod(skipna=False) - 1.0


def equity_curve(returns: PandasT, initial: float = 1.0) -> PandasT:
    """Equity index with leading warm-up gaps treated as zero returns."""
    starting_value = finite_real(initial, name="initial", minimum=0.0, strict=True)
    prepared = _prepare_compounding_returns(returns)
    return starting_value * (1.0 + prepared).cumprod(skipna=False)
