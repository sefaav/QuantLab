"""Direct tests for `quantlab.dashboard.explorer.shared_components`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantlab.dashboard.explorer.shared_components import (
    render_stop_loss_take_profit_illustration,
)


class _FakeColumn:
    def __init__(self, value: float) -> None:
        self._value = value

    def slider(self, *args: object, **kwargs: object) -> float:
        return self._value


class _FakeStreamlit:
    def __init__(
        self, stop_loss_pct: float = 0.1, take_profit_pct: float = 0.0
    ) -> None:
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct
        self.warnings: list[str] = []

    def markdown(self, *args: object, **kwargs: object) -> None:
        pass

    def caption(self, *args: object, **kwargs: object) -> None:
        pass

    def columns(self, n: int) -> list[_FakeColumn]:
        return [_FakeColumn(self._stop_loss_pct), _FakeColumn(self._take_profit_pct)]

    def warning(self, message: str, **kwargs: object) -> None:
        self.warnings.append(message)

    def dataframe(self, *args: object, **kwargs: object) -> None:
        pass


def _patch_render_price_chart(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out chart rendering -- these tests only care about the
    gap-detection/warning logic, not Plotly figure construction."""
    import quantlab.dashboard.explorer.shared_components as shared_components

    monkeypatch.setattr(
        shared_components,
        "render_price_chart",
        lambda *args, **kwargs: None,
    )


def test_internal_price_gap_while_held_triggers_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: `pct_change().fillna(0.0)` used to silently turn a
    genuine internal missing price into a flat 0% return, which could hide
    a real stop-loss/take-profit trigger. A NaN price mid-series while the
    position is nonzero must be reported via `st.warning`, naming the
    affected date(s)."""
    _patch_render_price_chart(monkeypatch)
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.Series([100.0, 101.0, np.nan, 103.0, 104.0, 105.0], index=idx)
    positions = pd.Series([0.0, 0.5, 0.5, 0.5, 0.5, 0.5], index=idx)

    st: Any = _FakeStreamlit(stop_loss_pct=0.1)
    render_stop_loss_take_profit_illustration(st, positions, prices, key_prefix="x")

    assert len(st.warnings) == 1
    assert "asset" in st.warnings[0]
    assert "2024-01-03" in st.warnings[0]  # the NaN price itself
    assert "2024-01-04" in st.warnings[0]  # pct_change's own next-day NaN


def test_no_warning_when_the_gap_coincides_with_a_flat_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing price while the position is flat (0) carries no risk of
    hiding a stop-loss/take-profit trigger -- must not warn."""
    _patch_render_price_chart(monkeypatch)
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.Series([100.0, 101.0, np.nan, 103.0, 104.0, 105.0], index=idx)
    positions = pd.Series([0.0, 0.0, 0.0, 0.0, 0.5, 0.5], index=idx)

    st: Any = _FakeStreamlit(stop_loss_pct=0.1)
    render_stop_loss_take_profit_illustration(st, positions, prices, key_prefix="x")

    assert st.warnings == []


def test_first_observation_is_zeroed_not_reported_as_a_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The very first observation has no prior price to compare against --
    a structural absence, not a genuine missing return -- and must be
    silently zeroed even when the position is already nonzero there,
    never reported as a warned gap."""
    _patch_render_price_chart(monkeypatch)
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx)
    positions = pd.Series([0.5, 0.5, 0.5, 0.5], index=idx)

    st: Any = _FakeStreamlit(stop_loss_pct=0.1)
    render_stop_loss_take_profit_illustration(st, positions, prices, key_prefix="x")

    assert st.warnings == []


def test_both_thresholds_disabled_returns_before_computing_returns_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check: with both thresholds at 0 (disabled), the function
    returns early and never touches the gap-detection path."""
    _patch_render_price_chart(monkeypatch)
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.Series([100.0, np.nan, 102.0], index=idx)
    positions = pd.Series([0.5, 0.5, 0.5], index=idx)

    st: Any = _FakeStreamlit(stop_loss_pct=0.0, take_profit_pct=0.0)
    render_stop_loss_take_profit_illustration(st, positions, prices, key_prefix="x")

    assert st.warnings == []
