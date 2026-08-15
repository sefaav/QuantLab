"""Deterministic cleaning for canonical OHLCV data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.config import MissingValuePolicy
from quantlab.constants import (
    ADJUSTED_CLOSE,
    CLOSE,
    HIGH,
    LOW,
    OHLCV_COLUMNS,
    OPEN,
    PRICE_COLUMNS,
    SYMBOL,
    TIMESTAMP,
    VOLUME,
)
from quantlab.exceptions import DataValidationError
from quantlab.logging_config import get_logger

logger = get_logger(__name__)

_FLOAT_COLUMNS = (OPEN, HIGH, LOW, CLOSE, ADJUSTED_CLOSE, VOLUME)


class DataCleaner:
    """Clean canonical long OHLCV data.

    Args:
        missing_value_policy: How to treat missing values across price columns.
        forward_fill_limit: Maximum consecutive bars to fill per symbol.
    """

    def __init__(
        self,
        missing_value_policy: MissingValuePolicy = MissingValuePolicy.DROP,
        forward_fill_limit: int = 1,
    ) -> None:
        if (
            isinstance(forward_fill_limit, bool)
            or not isinstance(forward_fill_limit, int)
            or forward_fill_limit <= 0
        ):
            raise ValueError("forward_fill_limit must be a positive integer.")
        self.missing_value_policy = MissingValuePolicy(missing_value_policy)
        self.forward_fill_limit = forward_fill_limit

    # ------------------------------------------------------------------ #
    # Individual operations do not mutate their input.
    # ------------------------------------------------------------------ #
    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate ``(timestamp, symbol)`` rows, keeping the last."""
        before = len(data)
        out = data.drop_duplicates(subset=[TIMESTAMP, SYMBOL], keep="last")
        removed = before - len(out)
        if removed:
            logger.info("Removed %d duplicate rows.", removed)
        return out

    def sort_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sort by symbol then timestamp so every symbol series is ordered."""
        return data.sort_values([SYMBOL, TIMESTAMP]).reset_index(drop=True)

    def remove_invalid_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with non-positive or non-finite prices, preserving NaN."""
        cols = [c for c in PRICE_COLUMNS if c in data.columns]
        if not cols:
            return data
        prices = data[cols]
        price_values = prices.to_numpy(dtype=float)
        finite_positive = pd.DataFrame(
            np.isfinite(price_values) & (price_values > 0),
            index=prices.index,
            columns=prices.columns,
        )
        valid = (finite_positive | prices.isna()).all(axis=1)
        removed = int((~valid).sum())
        if removed:
            logger.warning(
                "Removed %d rows with non-positive or non-finite prices.", removed
            )
        return data.loc[valid].reset_index(drop=True)

    def remove_non_finite_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows containing positive or negative infinity."""
        cols = [c for c in _FLOAT_COLUMNS if c in data.columns]
        if not cols:
            return data
        infinite = np.isinf(data[cols].to_numpy(dtype=float)).any(axis=1)
        removed = int(infinite.sum())
        if removed:
            logger.warning("Removed %d rows with infinite numeric values.", removed)
        return data.loc[~infinite].reset_index(drop=True)

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured missing-value policy.

        Policies:
            * ``drop``: drop rows containing any missing price value.
            * ``forward_fill``: fill price columns within each symbol, up to
              ``forward_fill_limit`` consecutive bars; never back-fill.
            * ``raise``: raise if any canonical value is missing.
            * ``none``: leave missing values untouched (caller must handle).

        Returns:
            The frame after applying the policy.

        Raises:
            DataValidationError: Under the ``raise`` policy when gaps exist.
        """
        policy = self.missing_value_policy
        if policy is MissingValuePolicy.NONE:
            return data

        if policy is MissingValuePolicy.RAISE:
            canonical_cols = [c for c in OHLCV_COLUMNS if c in data.columns]
            missing_counts = data[canonical_cols].isna().sum()
            missing_counts = missing_counts[missing_counts > 0]
            n_missing = int(missing_counts.sum())
            if n_missing == 0:
                return data
            raise DataValidationError(
                f"Found {n_missing} missing canonical values with policy "
                f"'raise': {missing_counts.astype(int).to_dict()}."
            )

        price_cols = [c for c in PRICE_COLUMNS if c in data.columns]
        n_missing = int(data[price_cols].isna().to_numpy().sum())
        if n_missing == 0:
            return data
        if policy is MissingValuePolicy.DROP:
            out = data.dropna(subset=price_cols).reset_index(drop=True)
            logger.info("Dropped %d rows with missing prices.", len(data) - len(out))
            return out
        if policy is MissingValuePolicy.FORWARD_FILL:
            out = data.sort_values([SYMBOL, TIMESTAMP]).copy()
            affected_rows = out[price_cols].isna().any(axis=1)
            out[price_cols] = out.groupby(SYMBOL, dropna=False)[price_cols].ffill(
                limit=self.forward_fill_limit
            )

            unresolved = out[price_cols].isna().any(axis=1)
            if unresolved.any():
                logger.warning(
                    "Dropped %d rows whose missing prices exceeded the "
                    "forward-fill limit.",
                    int(unresolved.sum()),
                )
                out = out.loc[~unresolved].copy()
                affected_rows = affected_rows.loc[out.index]

            ohlc = {OPEN, HIGH, LOW, CLOSE}
            if ohlc.issubset(out.columns):
                high_ok = out[HIGH] >= out[[OPEN, CLOSE, LOW]].max(axis=1)
                low_ok = out[LOW] <= out[[OPEN, CLOSE, HIGH]].min(axis=1)
                inconsistent = affected_rows & ~(high_ok & low_ok)
                if inconsistent.any():
                    logger.warning(
                        "Dropped %d forward-filled rows with inconsistent OHLC bounds.",
                        int(inconsistent.sum()),
                    )
                    out = out.loc[~inconsistent]

            logger.info(
                "Forward-filled missing prices with a %d-bar limit.",
                self.forward_fill_limit,
            )
            return out.reset_index(drop=True)
        return data  # pragma: no cover - exhaustive above

    def standardize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Coerce timestamp/symbol/float columns to canonical dtypes."""
        out = data.copy()
        # Canonical naive timestamps represent UTC; aware inputs are converted.
        out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP], utc=True).dt.tz_localize(None)
        out[SYMBOL] = out[SYMBOL].astype("string").astype(str)
        for col in _FLOAT_COLUMNS:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        return out

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run dtype, deduplication, ordering, missing-data, and value checks."""
        out = self.standardize_dtypes(data)
        out = self.remove_duplicates(out)
        out = self.sort_data(out)
        out = self.handle_missing_values(out)
        out = self.remove_non_finite_values(out)
        out = self.remove_invalid_prices(out)
        return out
