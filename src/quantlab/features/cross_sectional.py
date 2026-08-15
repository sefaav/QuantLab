"""Cross-sectional features computed independently at each date."""

from __future__ import annotations

import pandas as pd

from quantlab.constants import EPSILON
from quantlab.features._validation import finite_real, numeric_pandas


def _validate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(scores, pd.DataFrame):
        raise TypeError("scores must be a pandas DataFrame.")
    return numeric_pandas(scores, name="scores")


def cross_sectional_rank(
    scores: pd.DataFrame, *, ascending: bool = True
) -> pd.DataFrame:
    """Rank assets within each date, leaving missing scores unranked."""
    validated = _validate_scores(scores)
    return validated.rank(axis=1, ascending=ascending, na_option="keep")


def cross_sectional_percentile(scores: pd.DataFrame) -> pd.DataFrame:
    """Return percentile ranks in ``(0, 1]`` within each date."""
    validated = _validate_scores(scores)
    return validated.rank(axis=1, pct=True, na_option="keep")


def cross_sectional_zscore(scores: pd.DataFrame) -> pd.DataFrame:
    """Standardise each date's cross-section to zero mean and unit variance."""
    validated = _validate_scores(scores)
    mean = validated.mean(axis=1)
    std = validated.std(axis=1, ddof=1)
    return validated.sub(mean, axis=0).div(std + EPSILON, axis=0)


def cross_sectional_demean(scores: pd.DataFrame) -> pd.DataFrame:
    """Subtract each date's cross-sectional mean."""
    validated = _validate_scores(scores)
    return validated.sub(validated.mean(axis=1), axis=0)


def select_top_bottom(
    scores: pd.DataFrame,
    top_fraction: float,
    bottom_fraction: float = 0.0,
) -> pd.DataFrame:
    """Select disjoint top and bottom groups at each date.

    Positive fractions select at least one asset when the row contains data.
    Counts are rounded down otherwise. Ties are resolved deterministically by
    the original column order.
    """
    validated = _validate_scores(scores)
    top = finite_real(top_fraction, name="top_fraction", minimum=0.0)
    bottom = finite_real(bottom_fraction, name="bottom_fraction", minimum=0.0)
    if top > 1.0 or bottom > 1.0:
        raise ValueError("top_fraction and bottom_fraction must not exceed 1.")
    if top + bottom > 1.0:
        raise ValueError("top_fraction + bottom_fraction must not exceed 1.")

    selection = pd.DataFrame(
        0.0, index=validated.index, columns=validated.columns, dtype=float
    )
    for date, row in validated.iterrows():
        valid = row.dropna()
        top_k = _selection_count(len(valid), top)
        bottom_k = _selection_count(len(valid), bottom)
        if top_k + bottom_k > len(valid):
            raise ValueError(
                "The requested top and bottom fractions cannot form disjoint "
                f"groups when only {len(valid)} asset(s) are available at {date!r}."
            )

        top_assets = valid.sort_values(ascending=False, kind="stable").index[:top_k]
        remaining = valid.drop(index=top_assets)
        bottom_assets = remaining.sort_values(ascending=True, kind="stable").index[
            :bottom_k
        ]
        selection.loc[date, top_assets] = 1.0
        selection.loc[date, bottom_assets] = -1.0
    return selection


def _selection_count(n_valid: int, fraction: float) -> int:
    count = int(n_valid * fraction)
    if count == 0 and n_valid > 0 and fraction > 0:
        return 1
    return count
