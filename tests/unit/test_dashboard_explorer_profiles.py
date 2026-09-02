"""Structural guarantees for the Strategy Explorer content registry.

These are not content-quality tests (no automated check can judge whether
a markdown explanation is actually good) -- they verify the invariants the
approved plan named explicitly: every registered strategy has a profile,
every profile documents exactly the parameters its strategy accepts
(including structurally-injected ones like ``price_type``/
``periods_per_year``, never silently excluded), every markdown field is
non-empty, and the dispatch mechanism stays name-free outside the profile
files themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    import pandas as pd

import quantlab.dashboard.explorer.profiles  # noqa: F401  (registration side effect)
from quantlab.dashboard.explorer.profile import (
    StrategyProfile,
    available_profiles,
    get_profile,
)
from quantlab.strategies.base import available_strategies, strategy_parameter_names

_SRC = Path(__file__).resolve().parents[2] / "src" / "quantlab"

_MARKDOWN_FIELDS = (
    "overview_md",
    "economic_intuition_md",
    "mathematical_definition_md",
    "assumptions_md",
    "diagnostics_md",
    "interpretation_md",
    "limitations_md",
)


def test_every_registered_strategy_has_a_profile() -> None:
    missing = set(available_strategies()) - set(available_profiles())
    assert not missing, f"No profile registered for: {sorted(missing)}"


def test_available_profiles_are_all_real_strategies() -> None:
    """A profile registered under a name the strategy registry doesn't
    recognise would silently never be reachable from the gallery."""
    unknown = set(available_profiles()) - set(available_strategies())
    assert not unknown, f"Profile registered for unknown strategy: {sorted(unknown)}"


@pytest.mark.parametrize("strategy_name", available_strategies())
def test_profile_documents_exactly_the_strategy_constructor_parameters(
    strategy_name: str,
) -> None:
    """No parameter is missing, and none is documented that doesn't exist --
    including ``price_type``/``periods_per_year`` structurally injected by
    the runner, which must never be excluded just because they aren't
    passed explicitly in YAML."""
    profile = get_profile(strategy_name)
    assert profile is not None
    documented = {parameter.name for parameter in profile.parameters}
    expected = strategy_parameter_names(strategy_name)
    assert documented == expected, (
        f"{strategy_name}: documented={sorted(documented)} vs "
        f"expected={sorted(expected)}"
    )


@pytest.mark.parametrize("strategy_name", available_strategies())
def test_profile_markdown_fields_are_non_empty(strategy_name: str) -> None:
    profile = get_profile(strategy_name)
    assert profile is not None
    for field in _MARKDOWN_FIELDS:
        value = getattr(profile, field)
        assert isinstance(value, str)
        assert value.strip(), f"{strategy_name}.{field} is empty"


@pytest.mark.parametrize("strategy_name", available_strategies())
def test_profile_lab_is_callable(strategy_name: str) -> None:
    profile = get_profile(strategy_name)
    assert profile is not None
    assert callable(profile.lab)


@pytest.mark.parametrize("strategy_name", available_strategies())
def test_profile_display_name_and_category_are_set(strategy_name: str) -> None:
    profile = get_profile(strategy_name)
    assert profile is not None
    assert profile.display_name.strip()
    assert profile.category.strip()


def test_registered_profile_is_a_strategy_profile_instance() -> None:
    for name in available_profiles():
        assert isinstance(get_profile(name), StrategyProfile)


def test_html_report_never_names_a_specific_strategy() -> None:
    """`html_report.py`'s Robustness rendering dispatches a strategy's own
    results diagnostics generically, by `isinstance(value, DiagnosticsSection)`
    (see `_render_robustness`) -- never by strategy name. If a strategy name
    ever appears in this file, that architectural guarantee has been broken.
    """
    source = (_SRC / "reporting" / "html_report.py").read_text(encoding="utf-8")
    for strategy_name in available_strategies():
        assert strategy_name not in source, (
            f"html_report.py must not name '{strategy_name}' directly -- "
            "dispatch strategy-specific report content via "
            "quantlab.reporting.sections.DiagnosticsSection instead."
        )


def test_pairs_trading_diagnostics_respects_signal_price_type() -> None:
    """The pairs_trading profile's ``results_diagnostics.compute`` must price
    the diagnostics on whichever series the strategy itself actually trades
    (``strategy.signal_price_type`` -- ``strategy.parameters.price_type`` is
    rejected at config validation and can never override it) -- never
    silently default to adjusted_close regardless of what the config says,
    which would show a hedge ratio/spread that does not match what the
    backtest itself traded.
    """
    import numpy as np
    import pandas as pd

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

    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(0)
    close_a = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=60))
    close_b = 50.0 + 0.5 * (close_a - 100.0) + rng.normal(0.0, 0.1, size=60)
    # Deliberately offset from close, as if a corporate action had occurred,
    # so the two price choices produce numerically different diagnostics.
    adjusted_a = close_a * 0.5
    adjusted_b = close_b * 0.5

    def _frame(symbol: str, close: np.ndarray, adjusted: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                TIMESTAMP: idx,
                SYMBOL: symbol,
                OPEN: close,
                HIGH: close * 1.01,
                LOW: close * 0.99,
                CLOSE: close,
                ADJUSTED_CLOSE: adjusted,
                VOLUME: 1_000_000.0,
            }
        )

    data = pd.concat(
        [_frame("AAA", close_a, adjusted_a), _frame("BBB", close_b, adjusted_b)],
        ignore_index=True,
    )
    cfg_close = ExperimentConfig.from_dict(
        {
            "experiment_name": "price_type_test",
            "data": {
                "instruments": [
                    {"symbol": "AAA", "source": "csv", "calendar": "XNYS"},
                    {"symbol": "BBB", "source": "csv", "calendar": "XNYS"},
                ],
                "start_date": "2020-01-01",
                "end_date": "2020-04-01",
            },
            "portfolio": {"allocator": "signal_proportional"},
            "strategy": {
                "name": "pairs_trading",
                "signal_price_type": "close",
                "parameters": {
                    "symbol_a": "AAA",
                    "symbol_b": "BBB",
                    "formation_window": 20,
                    "indicator_window": 5,
                },
            },
        }
    )
    cfg_adjusted = cfg_close.revalidated_copy(
        update={
            "strategy": cfg_close.strategy.revalidated_copy(
                update={"signal_price_type": "adjusted_close"}
            )
        }
    )

    profile = get_profile("pairs_trading")
    assert profile is not None
    assert profile.results_diagnostics is not None
    diagnostics_close = profile.results_diagnostics.compute(data, cfg_close)
    diagnostics_adjusted = profile.results_diagnostics.compute(data, cfg_adjusted)

    close_spread = diagnostics_close.diagnostics.spread.dropna()
    adjusted_spread = diagnostics_adjusted.diagnostics.spread.dropna()
    assert not close_spread.equals(adjusted_spread)

    # Directly verify against a manual computation on the raw close prices,
    # rather than only checking "the two differ".
    from quantlab.data.base import price_matrix
    from quantlab.features.pairs_diagnostics import compute_pair_diagnostics

    expected_close = compute_pair_diagnostics(
        price_matrix(data, adjusted=False),
        "AAA",
        "BBB",
        formation_window=20,
        indicator_window=5,
        dynamic_hedge_ratio=True,
    )
    pd.testing.assert_series_equal(close_spread, expected_close.spread.dropna())


def _momentum_universe_data(n: int = 650) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

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

    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    frames = []
    for i, symbol in enumerate(["AAA", "BBB", "CCC"]):
        prices = 100.0 + np.cumsum(np.random.default_rng(i).normal(0.0, 1.0, size=n))
        prices = np.maximum(prices, 1.0)
        frames.append(
            pd.DataFrame(
                {
                    TIMESTAMP: idx,
                    SYMBOL: symbol,
                    OPEN: prices,
                    HIGH: prices * 1.01,
                    LOW: prices * 0.99,
                    CLOSE: prices,
                    ADJUSTED_CLOSE: prices,
                    VOLUME: 1_000_000.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("skip_period", [0, 21, 500])
def test_cross_sectional_momentum_diagnostics_handles_any_valid_skip_period(
    skip_period: int,
) -> None:
    """Regression test: the diagnostic's own forward-return horizon used
    to be fixed directly to `skip_period` -- `skip_period=0` is a
    perfectly valid strategy config (0-21 is even the documented typical
    range), but `holding_period=0` is rejected by
    `cross_sectional_momentum_persistence` (must be >= 1); a `skip_period`
    above 252 is also valid (only constrained to be < lookback_period) but
    would put the Results-tab slider's default value outside its own
    1-252 range. `compute()` must succeed for every valid skip_period,
    always defaulting the diagnostic's own horizon to a fixed,
    skip_period-independent value within [1, 252]."""
    from quantlab.config import ExperimentConfig

    lookback_period = 600 if skip_period >= 252 else 100
    data = _momentum_universe_data()
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "csmom_skip_period_test",
            "data": {
                "instruments": [
                    {"symbol": s, "source": "csv", "calendar": "XNYS"}
                    for s in ["AAA", "BBB", "CCC"]
                ],
                "start_date": "2019-01-01",
                "end_date": "2021-06-01",
            },
            "portfolio": {"allocator": "equal_weight"},
            "strategy": {
                "name": "cross_sectional_momentum",
                "parameters": {
                    "lookback_period": lookback_period,
                    "skip_period": skip_period,
                    "top_fraction": 0.5,
                },
            },
        }
    )
    profile = get_profile("cross_sectional_momentum")
    assert profile is not None
    assert profile.results_diagnostics is not None

    result = profile.results_diagnostics.compute(data, cfg)

    assert 1 <= result.holding_period <= 252
    assert result.skip_period == skip_period


@pytest.mark.parametrize("skip_period", [0, 21, 500])
def test_time_series_momentum_diagnostics_handles_any_valid_skip_period(
    skip_period: int,
) -> None:
    """Same regression as the cross-sectional case above, for
    time_series_momentum's own diagnostic."""
    from quantlab.config import ExperimentConfig

    lookback_period = 600 if skip_period >= 252 else 100
    data = _momentum_universe_data()
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "tsmom_skip_period_test",
            "data": {
                "instruments": [
                    {"symbol": s, "source": "csv", "calendar": "XNYS"}
                    for s in ["AAA", "BBB", "CCC"]
                ],
                "start_date": "2019-01-01",
                "end_date": "2021-06-01",
            },
            "portfolio": {"allocator": "equal_weight"},
            "strategy": {
                "name": "time_series_momentum",
                "parameters": {
                    "lookback_period": lookback_period,
                    "skip_period": skip_period,
                },
            },
        }
    )
    profile = get_profile("time_series_momentum")
    assert profile is not None
    assert profile.results_diagnostics is not None

    result = profile.results_diagnostics.compute(data, cfg)

    assert 1 <= result.holding_period <= 252
    assert result.skip_period == skip_period


@pytest.mark.parametrize("slow_window", [2, 3, 4])
def test_trend_following_results_er_slider_handles_a_slow_window_below_five(
    slow_window: int,
) -> None:
    """Regression test: `fast_window=1, slow_window=2` (and similarly
    small windows) are valid strategy configs -- only `fast_window <
    slow_window`, both `>= 1`, are enforced -- but the Results tab's
    Efficiency Ratio slider declares a fixed [5, 200] range. Its default
    value used to be `min(slow_window, 200)` directly, which could fall
    below 5 and put the slider's own default outside its declared bounds.
    Uses a real `AppTest` (not a fake streamlit stand-in) to exercise the
    actual widget construction, since the dashboard's own sidebar cannot
    reach a `slow_window` this small (its slider is bounded at 30) --
    the only way to observe this is a direct, config-driven scenario like
    a hand-written YAML config passed straight to the CLI."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    script = f"""
import pandas as pd
import streamlit as st
from quantlab.dashboard.explorer.profiles.trend_following import (
    TrendFollowingDiagnostics,
    _render_diagnostics,
)

prices = {{"AAA": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])}}
summary = pd.DataFrame(
    {{"Whipsaw (flips / 126p, latest)": [0.0], "Median Efficiency Ratio": [0.5]}},
    index=pd.Index(["AAA"], name="Symbol"),
)
result = TrendFollowingDiagnostics(
    summary=summary,
    prices=prices,
    fast_ma=prices,
    slow_ma=prices,
    signal=prices,
    slow_window={slow_window},
)
_render_diagnostics(st, result)
"""
    at = AppTest.from_string(script, default_timeout=30)
    at.run()

    assert not at.exception
    er_slider = next(s for s in at.slider if s.label == "Efficiency Ratio window")
    assert 5 <= cast(int, er_slider.value) <= 200


def _minimal_profile(**overrides: object) -> StrategyProfile:
    """Build an otherwise-valid StrategyProfile, letting a test override
    just the field(s) it wants to test the registration guard for."""
    fields: dict[str, object] = {
        "strategy_name": "buy_and_hold",
        "display_name": "Test",
        "category": "Test",
        "overview_md": "x",
        "economic_intuition_md": "x",
        "mathematical_definition_md": "x",
        "assumptions_md": "x",
        "diagnostics_md": "x",
        "interpretation_md": "x",
        "limitations_md": "x",
        "parameters": [],
        "lab": lambda st: None,
    }
    fields.update(overrides)
    return StrategyProfile(**fields)  # type: ignore[arg-type]


def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in a throwaway copy of the module-level registry so a test's
    own `register_profile()` calls can't leak into other tests -- restored
    automatically by monkeypatch's own teardown."""
    import quantlab.dashboard.explorer.profile as profile_module

    monkeypatch.setattr(profile_module, "_REGISTRY", dict(profile_module._REGISTRY))


@pytest.mark.parametrize(
    "field",
    [
        "overview_md",
        "economic_intuition_md",
        "mathematical_definition_md",
        "assumptions_md",
        "diagnostics_md",
        "interpretation_md",
        "limitations_md",
    ],
)
def test_register_profile_rejects_an_empty_markdown_field(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    from quantlab.dashboard.explorer.profile import register_profile

    _isolated_registry(monkeypatch)
    profile = _minimal_profile(**{field: "   "})
    with pytest.raises(ValueError, match=field):
        register_profile(profile, replace=True)


def test_register_profile_rejects_empty_display_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantlab.dashboard.explorer.profile import register_profile

    _isolated_registry(monkeypatch)
    with pytest.raises(ValueError, match="display_name"):
        register_profile(_minimal_profile(display_name=""), replace=True)


def test_register_profile_rejects_duplicate_parameter_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantlab.dashboard.explorer.profile import ParameterDoc, register_profile

    _isolated_registry(monkeypatch)
    duplicate = ParameterDoc(
        name="lookback_period",
        what="x",
        where="x",
        why="x",
        default="x",
        typical_range="x",
        effect_increase="x",
        effect_decrease="x",
        tradeoffs="x",
    )
    profile = _minimal_profile(parameters=[duplicate, duplicate])
    with pytest.raises(ValueError, match="duplicate"):
        register_profile(profile, replace=True)


@pytest.mark.parametrize(
    "field",
    [
        "name",
        "what",
        "where",
        "why",
        "default",
        "typical_range",
        "effect_increase",
        "effect_decrease",
        "tradeoffs",
    ],
)
def test_register_profile_rejects_an_empty_parameter_doc_field(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    from quantlab.dashboard.explorer.profile import ParameterDoc, register_profile

    _isolated_registry(monkeypatch)
    fields = {
        "name": "lookback_period",
        "what": "x",
        "where": "x",
        "why": "x",
        "default": "x",
        "typical_range": "x",
        "effect_increase": "x",
        "effect_decrease": "x",
        "tradeoffs": "x",
    }
    fields[field] = "   "
    profile = _minimal_profile(parameters=[ParameterDoc(**fields)])
    with pytest.raises(ValueError, match=field):
        register_profile(profile, replace=True)


def test_register_profile_allows_an_empty_parameter_doc_interactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`interactions` defaults to '' for a parameter with none to report --
    unlike every other ParameterDoc field, an empty one must not be rejected."""
    from quantlab.dashboard.explorer.profile import ParameterDoc, register_profile

    _isolated_registry(monkeypatch)
    parameter = ParameterDoc(
        name="lookback_period",
        what="x",
        where="x",
        why="x",
        default="x",
        typical_range="x",
        effect_increase="x",
        effect_decrease="x",
        tradeoffs="x",
        interactions="",
    )
    register_profile(_minimal_profile(parameters=[parameter]), replace=True)


def test_register_profile_rejects_an_unregistered_strategy_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantlab.dashboard.explorer.profile import register_profile

    _isolated_registry(monkeypatch)
    profile = _minimal_profile(strategy_name="not_a_real_strategy")
    with pytest.raises(ValueError, match="not_a_real_strategy"):
        register_profile(profile, replace=True)


def test_register_profile_rejects_a_non_callable_results_diagnostics_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantlab.dashboard.explorer.profile import ResultsDiagnostics, register_profile

    _isolated_registry(monkeypatch)
    diagnostics = ResultsDiagnostics(
        key="k",
        compute="not_callable",  # type: ignore[arg-type]
        render=lambda st, result: None,
        report_section=lambda result: None,  # type: ignore[arg-type,return-value]
    )
    profile = _minimal_profile(results_diagnostics=diagnostics)
    with pytest.raises(ValueError, match="compute"):
        register_profile(profile, replace=True)


def test_register_profile_rejects_an_empty_results_diagnostics_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantlab.dashboard.explorer.profile import ResultsDiagnostics, register_profile

    _isolated_registry(monkeypatch)
    diagnostics = ResultsDiagnostics(
        key="   ",
        compute=lambda data, cfg: None,
        render=lambda st, result: None,
        report_section=lambda result: None,  # type: ignore[arg-type,return-value]
    )
    profile = _minimal_profile(results_diagnostics=diagnostics)
    with pytest.raises(ValueError, match="key"):
        register_profile(profile, replace=True)


def test_register_profile_rejects_a_colliding_results_diagnostics_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`results_diagnostics.key` doubles as the robustness-dict/session_state
    key -- a collision between two profiles would let them silently clobber
    each other's diagnostics."""
    from quantlab.dashboard.explorer.profile import ResultsDiagnostics, register_profile

    _isolated_registry(monkeypatch)
    diagnostics = ResultsDiagnostics(
        key="shared_key",
        compute=lambda data, cfg: None,
        render=lambda st, result: None,
        report_section=lambda result: None,  # type: ignore[arg-type,return-value]
    )
    first = _minimal_profile(
        strategy_name="buy_and_hold", results_diagnostics=diagnostics
    )
    second = _minimal_profile(
        strategy_name="mean_reversion", results_diagnostics=diagnostics
    )
    register_profile(first, replace=True)
    with pytest.raises(ValueError, match="shared_key"):
        register_profile(second, replace=True)


def test_gallery_and_detail_pages_never_name_a_specific_strategy() -> None:
    """The gallery/detail pages are driven entirely by `available_strategies()`
    / `get_profile()` -- a strategy name appearing in either file would mean
    a new strategy needs a dashboard code change beyond its own profile file,
    which is exactly what the registry pattern exists to avoid."""
    explorer_dir = _SRC / "dashboard" / "explorer"
    for filename in ("gallery.py", "detail.py"):
        source = (explorer_dir / filename).read_text(encoding="utf-8")
        for strategy_name in available_strategies():
            assert strategy_name not in source, (
                f"{filename} must not name '{strategy_name}' directly."
            )
