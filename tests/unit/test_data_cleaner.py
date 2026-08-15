"""Tests for data cleaning and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_ohlcv

from quantlab.config import MissingValuePolicy
from quantlab.constants import CLOSE, HIGH, LOW, SYMBOL, TIMESTAMP, VOLUME
from quantlab.data.base import pivot_field, price_matrix
from quantlab.data.cleaner import DataCleaner
from quantlab.data.validator import DataValidator
from quantlab.exceptions import DataValidationError


def _simple(symbol: str = "AAA") -> pd.DataFrame:
    return make_ohlcv(symbol, np.array([100.0, 101.0, 102.0, 103.0, 104.0]))


def test_remove_duplicates() -> None:
    data = _simple()
    dup = pd.concat([data, data.iloc[[2]]], ignore_index=True)
    cleaned = DataCleaner().remove_duplicates(dup)
    assert len(cleaned) == len(data)


def test_sort_data() -> None:
    data = _simple().sample(frac=1.0, random_state=0).reset_index(drop=True)
    cleaned = DataCleaner().sort_data(data)
    assert cleaned[TIMESTAMP].is_monotonic_increasing


def test_remove_invalid_prices() -> None:
    data = _simple()
    data.loc[1, CLOSE] = -5.0  # impossible price
    cleaned = DataCleaner().remove_invalid_prices(data)
    assert len(cleaned) == len(data) - 1
    assert (cleaned[CLOSE] > 0).all()


def test_remove_invalid_prices_rejects_positive_infinity() -> None:
    data = _simple()
    data.loc[1, CLOSE] = np.inf
    cleaned = DataCleaner().remove_invalid_prices(data)
    assert len(cleaned) == len(data) - 1
    assert np.isfinite(cleaned[CLOSE]).all()


def test_clean_rejects_infinite_volume() -> None:
    data = _simple()
    data.loc[1, VOLUME] = np.inf
    cleaned = DataCleaner().clean(data)
    assert len(cleaned) == len(data) - 1
    assert np.isfinite(cleaned[VOLUME]).all()


def test_missing_policy_drop() -> None:
    data = _simple()
    data.loc[2, CLOSE] = np.nan
    cleaner = DataCleaner(MissingValuePolicy.DROP)
    cleaned = cleaner.handle_missing_values(data)
    assert len(cleaned) == len(data) - 1


def test_missing_policy_raise() -> None:
    data = _simple()
    data.loc[2, CLOSE] = np.nan
    cleaner = DataCleaner(MissingValuePolicy.RAISE)
    with pytest.raises(DataValidationError):
        cleaner.handle_missing_values(data)


def test_missing_policy_raise_includes_volume() -> None:
    data = _simple()
    data.loc[2, VOLUME] = np.nan
    cleaner = DataCleaner(MissingValuePolicy.RAISE)
    with pytest.raises(DataValidationError, match="volume"):
        cleaner.handle_missing_values(data)


def test_missing_policy_forward_fill_no_backfill() -> None:
    data = _simple()
    data.loc[2, CLOSE] = np.nan
    cleaner = DataCleaner(MissingValuePolicy.FORWARD_FILL)
    cleaned = cleaner.handle_missing_values(data)
    # The gap is filled with the *previous* close (101), never a future value.
    assert cleaned.loc[cleaned[TIMESTAMP] == data.loc[2, TIMESTAMP], CLOSE].iloc[
        0
    ] == pytest.approx(101.0)


def test_forward_fill_is_limited_to_one_consecutive_bar_by_default() -> None:
    data = _simple()
    data.loc[[2, 3], CLOSE] = np.nan
    cleaned = DataCleaner(MissingValuePolicy.FORWARD_FILL).handle_missing_values(data)
    assert len(cleaned) == len(data) - 1
    assert cleaned.loc[cleaned[TIMESTAMP] == data.loc[2, TIMESTAMP], CLOSE].iloc[
        0
    ] == pytest.approx(101.0)
    assert data.loc[3, TIMESTAMP] not in set(cleaned[TIMESTAMP])


def test_forward_fill_drops_newly_inconsistent_ohlc_row() -> None:
    data = _simple()
    data.loc[2, HIGH] = np.nan
    data.loc[2, CLOSE] = 200.0
    cleaned = DataCleaner(MissingValuePolicy.FORWARD_FILL).handle_missing_values(data)
    assert data.loc[2, TIMESTAMP] not in set(cleaned[TIMESTAMP])


def test_missing_policy_none_leaves_gaps() -> None:
    data = _simple()
    data.loc[2, CLOSE] = np.nan
    cleaner = DataCleaner(MissingValuePolicy.NONE)
    cleaned = cleaner.handle_missing_values(data)
    assert cleaned[CLOSE].isna().sum() == 1


def test_validator_detects_duplicates_strict() -> None:
    data = _simple("SPY")
    dup = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        DataValidator().validate(dup, strict=True)


def test_validator_detects_negative_price_strict() -> None:
    data = _simple()
    data.loc[1, CLOSE] = -1.0
    with pytest.raises(DataValidationError, match="non-positive"):
        DataValidator().validate(data, strict=True)


def test_validator_detects_high_below_low() -> None:
    data = _simple()
    # Force an impossible bar: high < low.
    data.loc[2, HIGH] = 50.0
    data.loc[2, LOW] = 200.0
    report = DataValidator().validate(data, strict=False)
    assert report.invalid_price_count > 0
    assert any("OHLC" in w for w in report.warnings)


def test_validator_report_counts_missing_values() -> None:
    data = _simple()
    data.loc[1, CLOSE] = np.nan
    report = DataValidator().validate(data, strict=False)
    assert report.missing_value_count.get(CLOSE) == 1
    assert report.row_count == len(data)


def test_validator_flags_symbol_absent_via_empty() -> None:
    empty = _simple().iloc[0:0]
    report = DataValidator().validate(empty, strict=False)
    assert any("empty" in w.lower() for w in report.warnings)


def test_pivot_and_price_matrix(synthetic_panel: pd.DataFrame) -> None:
    wide = price_matrix(synthetic_panel)
    assert list(wide.columns) == ["AAA", "BBB", "CCC"]
    assert wide.index.is_monotonic_increasing
    # No reshaping should introduce NaNs for a complete panel.
    assert not wide.isna().any().any()
    close_wide = pivot_field(synthetic_panel, CLOSE)
    assert close_wide.shape == wide.shape


def test_pivot_rejects_duplicate_timestamp_symbol_pairs() -> None:
    data = _simple()
    duplicate = data.iloc[[0]].copy()
    duplicate.loc[:, CLOSE] = 999.0
    with pytest.raises(DataValidationError, match="duplicate"):
        pivot_field(pd.concat([data, duplicate], ignore_index=True), CLOSE)


def test_full_clean_pipeline_roundtrip(synthetic_panel: pd.DataFrame) -> None:
    messy = pd.concat(
        [synthetic_panel, synthetic_panel.iloc[[0, 1, 2]]], ignore_index=True
    )
    cleaned = DataCleaner(MissingValuePolicy.DROP).clean(messy)
    # Duplicates removed and every symbol sorted.
    assert not cleaned.duplicated(subset=[TIMESTAMP, SYMBOL]).any()
    for _, g in cleaned.groupby(SYMBOL):
        assert g[TIMESTAMP].is_monotonic_increasing
