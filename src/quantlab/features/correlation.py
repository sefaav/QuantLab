"""Multi-asset correlation diagnostics."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from quantlab.features._validation import numeric_pandas

_CorrelationMethod = Literal["pearson", "kendall", "spearman"]


def correlation_matrix(
    prices: pd.DataFrame, *, method: _CorrelationMethod = "pearson"
) -> pd.DataFrame:
    """Return the symbol x symbol correlation matrix of simple returns.

    Computed on returns, not raw price levels -- price-level correlation is
    routinely inflated by a shared trend even between economically
    unrelated assets, while return correlation reflects actual co-movement.
    ``method`` is passed straight through to ``DataFrame.corr``.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    validated = numeric_pandas(prices, name="prices", strictly_positive=True)
    if method not in {"pearson", "kendall", "spearman"}:
        raise ValueError(
            f"method must be one of 'pearson'/'kendall'/'spearman', got {method!r}."
        )
    returns = validated.pct_change(fill_method=None)
    return returns.corr(method=method)
