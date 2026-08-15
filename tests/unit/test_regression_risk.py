"""Regression tests for risk behavior."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from tests.conftest import geometric_series, make_ohlcv
from tests.regression_helpers import (
    _base_config_dict,
    _flat_execution_model,
    _hourly_symbol_frame,
    _minimal_ohlcv_frame,
    _rf_test_setup,
)

from quantlab.config import ExperimentConfig
from quantlab.exceptions import InvalidConfigurationError


def test_equity_from_returns_includes_first_return() -> None:
    from quantlab.risk.metrics import equity_from_returns, total_return

    returns = pd.Series([0.10, 0.0])
    equity = equity_from_returns(returns)
    # True compounded return over both periods is +10%, not 0%.
    assert total_return(equity) == pytest.approx(0.10)


def test_yearly_table_counts_first_day_of_each_year() -> None:
    from quantlab.reporting.tables import subperiod_table

    idx = pd.date_range("2020-01-01", periods=500, freq="D")
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.005, len(idx)), index=idx)

    class _FakeResult:
        returns: pd.Series
        turnover: pd.Series
        trades: pd.DataFrame
        config: Any

    result = _FakeResult()
    result.returns = returns
    result.turnover = pd.Series(0.0, index=idx)
    result.trades = pd.DataFrame(columns=["timestamp"])
    fake_backtest = type("Backtest", (), {"risk_free_rate": 0.0})()
    result.config = type(
        "Cfg", (), {"periods_per_year": 252, "backtest": fake_backtest}
    )()
    table = subperiod_table(result)  # type: ignore[arg-type]
    # Manually compute year 2021's true compounded return (first day included).
    year_2021 = returns.loc["2021-01-01":"2021-12-31"].to_numpy()
    expected = float(np.prod(1.0 + year_2021) - 1.0)
    row = table.loc[table["Period"] == "2021", "Return"].iloc[0]
    assert row == pytest.approx(expected)


def test_static_hedge_ratio_no_lookahead() -> None:
    from quantlab.strategies.pairs_trading import _rolling_hedge_ratio

    a = pd.Series(np.linspace(100.0, 110.0, 20))
    b = pd.Series(np.linspace(50.0, 55.0, 20))
    beta = _rolling_hedge_ratio(a, b, window=5, dynamic=False)
    assert beta.iloc[:5].isna().all(), "beta must be undefined before formation ends"
    assert beta.iloc[5:].notna().all(), "beta must be defined from the window onward"

    # Changing future data must not change the beta value seen at earlier dates.
    b_changed_future = b.copy()
    b_changed_future.iloc[15:] += 1000.0
    beta_changed = _rolling_hedge_ratio(a, b_changed_future, window=5, dynamic=False)
    pd.testing.assert_series_equal(beta.iloc[:15], beta_changed.iloc[:15])


def test_slice_between_respects_both_bounds() -> None:
    from quantlab.validation.walk_forward import _slice_between

    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    data = pd.DataFrame({"timestamp": idx, "symbol": "AAA", "close": 1.0})
    sliced = _slice_between(data, idx[30], idx[60])
    ts = pd.to_datetime(sliced["timestamp"])
    assert ts.min() == idx[30]
    assert ts.max() == idx[60]


def test_renormalize_within_cap_never_exceeds_cap() -> None:
    from quantlab.portfolio.position_sizing import renormalize_within_cap

    weights = pd.DataFrame({"A": [0.9], "B": [0.05], "C": [0.05]})
    capped = weights.clip(-0.30, 0.30)
    fixed = renormalize_within_cap(capped, target_gross=1.0, cap=0.30)
    assert fixed.iloc[0].max() <= 0.30 + 1e-9


def test_overlapping_top_bottom_fractions_rejected() -> None:
    from quantlab.features.cross_sectional import select_top_bottom

    df = pd.DataFrame({"A": [4.0], "B": [3.0], "C": [2.0], "D": [1.0]})
    with pytest.raises(ValueError, match="must not exceed 1"):
        select_top_bottom(df, top_fraction=0.75, bottom_fraction=0.75)


def test_slice_range_excludes_day_after_end() -> None:
    from quantlab.data.loader import DataLoader

    data = make_ohlcv("AAA", np.linspace(100, 110, 5), start="2020-01-01", freq="D")
    sliced = DataLoader._slice_range(
        data, date(2020, 1, 1), date(2020, 1, 2), "1d", is_247_market=False
    )
    dates = sliced["timestamp"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == ["2020-01-01", "2020-01-02"]


def test_benchmark_outside_universe_is_loaded_but_not_tradable(
    tmp_path: Path,
) -> None:
    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.data.loader import DataLoader
    from quantlab.data.storage import ParquetStorage

    raw = tmp_path / "raw"
    raw.mkdir()
    for sym, seed in [("EWA", 1), ("EWC", 2), ("SPY", 3)]:
        p = geometric_series(200, mu=0.0004, sigma=0.01, s0=100.0, seed=seed)
        make_ohlcv(sym, p, start="2019-01-01").to_csv(raw / f"{sym}.csv", index=False)

    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "bench_test",
            "data": {
                "source": "csv",
                "market_calendar": "XNYS",
                "symbols": ["EWA", "EWC"],
                "start_date": "2019-01-01",
                "end_date": "2019-08-01",
            },
            "strategy": {
                "name": "pairs_trading",
                "parameters": {
                    "symbol_a": "EWA",
                    "symbol_b": "EWC",
                    "formation_window": 40,
                    "zscore_window": 15,
                },
            },
            "portfolio": {"allocator": "signal_proportional"},
            "execution": {
                "commission_bps": 2.0,
                "spread_bps": 3.0,
                "slippage_bps": 2.0,
            },
            "backtest": {"initial_capital": 100_000, "benchmark_symbol": "SPY"},
        }
    )
    loader = DataLoader(
        storage=ParquetStorage(cache_dir=tmp_path / "c", metadata_dir=tmp_path / "m"),
        raw_dir=raw,
    )
    data, _ = loader.load(cfg)
    assert "SPY" in data["symbol"].unique()

    result = run_backtest_from_config(data, cfg)
    assert result.benchmark_returns is not None
    assert "SPY" not in result.signals.columns
    assert "SPY" not in result.weights.columns


def test_unknown_validation_method_rejected() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "symbols": ["A"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {"name": "buy_and_hold"},
                "validation": {"method": "typo"},
            }
        )


def test_unknown_optimization_metric_rejected_at_config_load() -> None:
    with pytest.raises(InvalidConfigurationError):
        ExperimentConfig.from_dict(
            {
                "experiment_name": "x",
                "data": {
                    "symbols": ["A"],
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                },
                "strategy": {"name": "buy_and_hold"},
                "validation": {"optimization_metric": "sharp_typo"},
            }
        )


def test_crypto_monthly_annualises_at_12_not_252() -> None:
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "x",
            "data": {
                "source": "csv",
                "symbols": ["BTC"],
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "frequency": "1mo",
                "market_calendar": "24/7",
            },
            "strategy": {"name": "buy_and_hold"},
        }
    )
    assert cfg.data.is_247_market
    assert cfg.periods_per_year == 12


def test_subperiod_table_sharpe_matches_engine_sharpe() -> None:
    from quantlab.backtesting.runner import run_backtest_from_config
    from quantlab.reporting.tables import subperiod_table

    data, cfg = _rf_test_setup()
    result = run_backtest_from_config(data, cfg)
    table = subperiod_table(result)
    full_sample_sharpe = table.loc[table["Period"] == "Full sample", "Sharpe"].iloc[0]
    assert full_sample_sharpe == pytest.approx(result.metrics["sharpe_ratio"])


def test_genuinely_invalid_price_still_removed() -> None:
    """NaN tolerance in the missing-value policy must not let a real
    non-positive price through."""
    from quantlab.config import MissingValuePolicy
    from quantlab.data.cleaner import DataCleaner

    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    data = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": "AAA",
            "open": [100.0, -5.0, 103.0],
            "high": [101.0, -4.0, 104.0],
            "low": [99.0, -6.0, 102.0],
            "close": [100.5, -5.0, 103.5],
            "adjusted_close": [100.5, -5.0, 103.5],
            "volume": [1000.0] * 3,
        }
    )
    out = DataCleaner(MissingValuePolicy.DROP).clean(data)
    assert len(out) == 2
    assert (out["close"] > 0).all()


def test_2x_hourly_mismatch_now_flagged() -> None:
    from quantlab.data.validator import DataValidator

    frame = _hourly_symbol_frame("2h", 40, "BTCUSDT")
    report = DataValidator(
        expected_frequency="1h", min_coverage_rows=5, is_247_market=True
    ).validate(frame)
    assert any("does not match the declared frequency" in w for w in report.warnings)


def test_2x_daily_mismatch_now_flagged() -> None:
    from quantlab.data.validator import DataValidator

    frame = _hourly_symbol_frame("2D", 40, "AAA")
    report = DataValidator(
        expected_frequency="1d", min_coverage_rows=5, is_247_market=False
    ).validate(frame)
    assert any("does not match the declared frequency" in w for w in report.warnings)


def test_end_boundary_is_strictly_exclusive() -> None:
    from datetime import date

    from quantlab.data.validator import DataValidator

    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2020-01-30"),
                pd.Timestamp("2020-01-31"),
                pd.Timestamp("2020-02-01 00:00:00"),
            ],
            "symbol": "AAA",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adjusted_close": 100.0,
            "volume": 1000.0,
        }
    )
    report = DataValidator(min_coverage_rows=1).validate(frame, end=date(2020, 1, 31))
    assert any("rows after requested end" in w for w in report.warnings)


def test_equity_never_goes_negative_and_does_not_resurrect() -> None:
    from quantlab.backtesting.accounting import run_accounting

    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    held = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    # A total loss is followed by a large rebound. Once equity reaches zero,
    # accounting must stop instead of letting the later gain resurrect it.
    asset_returns = pd.DataFrame({"A": [0.0, -1.0, 0.0, 1.0]}, index=idx)

    result = run_accounting(held, asset_returns, _flat_execution_model(), 100.0)
    assert (result.equity >= 0.0).all()
    assert result.equity.tolist() == [100.0, 0.0, 0.0, 0.0]
    assert (result.net_returns >= -1.0).all()


def test_ordinary_returns_unaffected_by_floor() -> None:
    """The -100% floor must be a no-op for any realistic (non-insolvent)
    return path — it should only ever bind on the pathological cases above."""
    from quantlab.backtesting.accounting import run_accounting

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    held = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=idx)
    asset_returns = pd.DataFrame({"A": [0.0, 0.02, -0.03, 0.01, -0.015]}, index=idx)
    result = run_accounting(held, asset_returns, _flat_execution_model(), 100.0)
    expected_equity = 100.0 * (1.0 + asset_returns["A"].fillna(0.0)).cumprod()
    # First period's contribution is 0 (weights start at 0 via shift(1)).
    expected_equity.iloc[0] = 100.0
    pd.testing.assert_series_equal(
        result.equity, expected_equity, check_names=False, check_freq=False
    )


def test_config_model_copy_revalidates_against_nan_and_infinity() -> None:
    from quantlab.validation.walk_forward import _with_params

    cfg = ExperimentConfig.from_yaml(
        Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    )
    with pytest.raises(InvalidConfigurationError, match="NaN/Infinity"):
        _with_params(cfg, {"threshold": float("nan")})
    with pytest.raises(ValidationError, match="NaN/Infinity"):
        cfg.execution.revalidated_copy(update={"commission_bps": float("inf")})
    # Sanity: an ordinary, valid copy still works.
    ok = cfg.execution.revalidated_copy(update={"commission_bps": 5.0})
    assert ok.commission_bps == 5.0


def test_revalidated_copy_rejects_what_plain_model_copy_would_silently_accept() -> None:
    from quantlab.config import BacktestConfig

    # `risk_free_rate` has no `Field(gt=..., ge=...)` bound at all -- the
    # exact case `_reject_non_json_safe`'s own docstring points to, where a
    # bound "happening" to reject NaN as a comparison side effect can't mask
    # whether the JSON-safe guard itself is doing the rejecting.
    cfg = BacktestConfig()
    with pytest.raises(ValidationError, match="NaN/Infinity"):
        cfg.revalidated_copy(update={"risk_free_rate": float("nan")})
    # Plain `model_copy` -- pydantic's own, un-overridden method -- applies
    # the same update with no validation at all.
    spliced = cfg.model_copy(update={"risk_free_rate": float("nan")})
    assert math.isnan(spliced.risk_free_rate)


def test_water_filling_converges_at_knife_edge_capacity() -> None:
    from quantlab.portfolio.position_sizing import renormalize_within_cap

    rng = np.random.default_rng(0)
    n = 100
    cap = 0.01
    target = n * cap
    # A skewed pre-cap distribution (power-law-ish) is what forces the
    # water-filling loop to freeze assets onto the cap one at a time across
    # many iterations, rather than most of them jumping to the cap at once.
    raw = rng.uniform(0, 1, n) ** 6
    raw = raw / raw.sum() * target
    clipped = np.clip(raw, -cap, cap)
    idx = pd.date_range("2020-01-01", periods=1)
    weights = pd.DataFrame([clipped], index=idx, columns=[f"S{i}" for i in range(n)])

    out = renormalize_within_cap(weights, target_gross=target, cap=cap)
    assert out.abs().sum(axis=1).iloc[0] == pytest.approx(target)
    assert (out.abs() <= cap + 1e-9).all().all()


def test_project_root_falls_back_to_home_when_not_a_dev_checkout(
    tmp_path: Path,
) -> None:
    import shutil
    import sys

    from quantlab import constants as real_constants

    package_dir = Path(real_constants.__file__).resolve().parent
    fake_site_packages = tmp_path / "site-packages"
    fake_site_packages.mkdir()
    shutil.copytree(package_dir, fake_site_packages / "quantlab")

    sys.path.insert(0, str(fake_site_packages))
    saved_modules = {
        name: mod
        for name, mod in list(sys.modules.items())
        if name == "quantlab" or name.startswith("quantlab.")
    }
    for name in saved_modules:
        del sys.modules[name]
    try:
        import quantlab.constants as fake_constants

        assert Path.home() / ".quantlab" == fake_constants.PROJECT_ROOT
        assert fake_constants.DATA_DIR == fake_constants.PROJECT_ROOT / "data"
    finally:
        sys.path.remove(str(fake_site_packages))
        for name in list(sys.modules):
            if name == "quantlab" or name.startswith("quantlab."):
                del sys.modules[name]
        sys.modules.update(saved_modules)


def test_source_hash_ttl_catches_a_mtime_preserving_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import shutil
    import time

    from quantlab.backtesting import engine

    real_root = Path(engine.__file__).resolve().parents[1]
    fake_root = tmp_path / "quantlab"
    shutil.copytree(real_root, fake_root)
    monkeypatch.setattr(
        engine, "__file__", str(fake_root / "backtesting" / "engine.py")
    )

    original = engine._source_hash()

    # Edit a file's content but restore its *original* mtime afterward --
    # exactly what a sync tool preserving the source machine's mtime does.
    target = fake_root / "constants.py"
    original_stat = target.stat()
    target.write_text(
        target.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8"
    )
    os.utime(target, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    # Immediately after, within the TTL window, the fingerprint alone can't
    # tell -- this is the accepted, *bounded* staleness, not a bug.
    still_stale = engine._source_hash()
    assert still_stale == original

    # Once the TTL has elapsed, the edit must be caught even though the
    # fingerprint never changed.
    monkeypatch.setattr(engine, "_source_hash_computed_at", time.monotonic() - 61.0)
    mutated = engine._source_hash()
    assert mutated != original


def test_grid_combinations_rejects_empty_value_list() -> None:
    from quantlab.exceptions import InvalidConfigurationError
    from quantlab.validation.walk_forward import _grid_combinations

    assert _grid_combinations({}) == [{}]  # a genuinely empty grid is fine
    with pytest.raises(InvalidConfigurationError, match="lookback_period"):
        _grid_combinations({"lookback_period": [], "skip_period": [0, 21]})


def test_experiment_name_path_traversal_rejected() -> None:
    payload = _base_config_dict()
    payload["experiment_name"] = "../../../evil"
    with pytest.raises(InvalidConfigurationError, match="Invalid experiment_name"):
        ExperimentConfig.from_dict(payload)


def test_dotted_ticker_and_underscored_name_are_accepted() -> None:
    """Path-component validation must not reject legitimate tickers/names —
    share-class tickers commonly use a dot (e.g. `BRK.B`), and experiment
    names commonly use underscores."""
    payload = _base_config_dict()
    payload["data"] = dict(payload["data"])
    payload["data"]["symbols"] = ["BRK.B"]
    payload["experiment_name"] = "my_experiment-2024.v1"
    cfg = ExperimentConfig.from_dict(payload)
    assert cfg.symbols == ["BRK.B"]
    assert cfg.experiment_name == "my_experiment-2024.v1"


def test_all_shipped_configs_pass_path_component_validation() -> None:
    """All shipped configs' symbols/experiment_name must pass path-component
    validation without making those rules too
    strict for real, in-repo configs."""
    import pathlib

    configs_dir = pathlib.Path(__file__).resolve().parents[2] / "configs"
    paths = sorted(configs_dir.glob("*.yaml"))
    assert paths, "expected at least one shipped YAML configuration"
    for path in paths:
        ExperimentConfig.from_yaml(path)  # must not raise


def test_rolling_sharpe_constant_window_is_zero_not_infinity() -> None:
    from quantlab.risk.metrics import rolling_sharpe_ratio

    returns = pd.Series([0.001] * 20)
    out = rolling_sharpe_ratio(returns, window=5)
    assert not np.isinf(out.to_numpy()).any()
    assert out.iloc[:4].isna().all()
    assert (out.iloc[4:] == 0.0).all()


def test_safe_leaves_already_safe_tokens_unchanged() -> None:
    from quantlab.data.storage import _safe

    for token in ["AAPL", "BRK-B", "BF.B", "binance", "1d"]:
        assert _safe(token) == token


def test_safe_is_deterministic_for_the_same_unsafe_token() -> None:
    """The hash-suffixed encoding must be stable across calls — the cache
    would otherwise never find a symbol's own previously-written file."""
    from quantlab.data.storage import _safe

    assert _safe("^GSPC") == _safe("^GSPC")


def test_safe_hash_suffixed_output_does_not_collide_with_a_raw_token() -> None:
    from quantlab.data.storage import _safe

    hashed = _safe("A=80")
    assert hashed == "A_80-5673847950"
    assert _safe(hashed) != hashed


def test_safe_prevents_hash_shape_collision_end_to_end(tmp_path: Path) -> None:
    """Two distinct symbols — one
    needing sanitisation, one that coincidentally looks like the first one's
    hash-suffixed output — must round-trip independently through the real
    cache."""
    from quantlab.data.storage import ParquetStorage

    storage = ParquetStorage(cache_dir=tmp_path / "cache", metadata_dir=tmp_path / "md")
    first = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=3),
            "symbol": "A=80",
            "open": 1.1,
            "high": 1.1,
            "low": 1.1,
            "close": 1.1,
            "adjusted_close": 1.1,
            "volume": 0.0,
        }
    )
    second = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=3),
            "symbol": "A_80-5673847950",
            "open": 2.2,
            "high": 2.2,
            "low": 2.2,
            "close": 2.2,
            "adjusted_close": 2.2,
            "volume": 0.0,
        }
    )
    storage.write_symbol(first, "yahoo", "A=80", "1d")
    storage.write_symbol(second, "yahoo", "A_80-5673847950", "1d")

    read_first = storage.read_symbol("yahoo", "A=80", "1d")
    read_second = storage.read_symbol("yahoo", "A_80-5673847950", "1d")
    assert read_first is not None
    assert read_second is not None
    assert (read_first["close"] == 1.1).all()
    assert (read_second["close"] == 2.2).all()


def test_safe_no_collisions_across_a_random_fuzz_sweep() -> None:
    import random
    import string

    from quantlab.data.storage import _safe

    rng = random.Random(0)
    charset = string.ascii_uppercase + string.digits + "=^-_. "
    tokens = {
        "".join(rng.choice(charset) for _ in range(rng.randint(1, 12)))
        for _ in range(5000)
    }
    tokens.add("A=80")
    tokens.add(_safe("A=80"))

    mapping: dict[str, str] = {}
    collisions = []
    for token in tokens:
        out = _safe(token)
        if out in mapping and mapping[out] != token:
            collisions.append((mapping[out], token, out))
        mapping[out] = token
    assert not collisions


def test_canonical_schema_converts_mixed_offsets_to_utc_before_stripping() -> None:
    from quantlab.data.base import ensure_canonical_schema

    frame = _minimal_ohlcv_frame(["2020-01-01T00:00:00+02:00", "2020-01-01T23:00:00Z"])
    out = ensure_canonical_schema(frame)
    assert out["timestamp"].dt.tz is None
    assert out["timestamp"].iloc[0] == pd.Timestamp("2019-12-31 22:00:00")
    assert out["timestamp"].iloc[1] == pd.Timestamp("2020-01-01 23:00:00")


def test_requirements_txt_streamlit_floor_matches_pyproject() -> None:
    import re
    import tomllib

    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    pyproject_floors: dict[str, str] = {}
    for spec in [
        *project["dependencies"],
        *project["optional-dependencies"]["dashboard"],
        *project["optional-dependencies"]["yahoo"],
    ]:
        match = re.match(r"^([A-Za-z0-9_.-]+)>=([0-9][0-9A-Za-z.]*)", spec)
        assert match, f"unexpected dependency spec format: {spec!r}"
        pyproject_floors[match.group(1).lower()] = match.group(2)

    requirements_floors: dict[str, str] = {}
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)>=([0-9][0-9A-Za-z.]*)", stripped)
        assert match, f"unexpected requirements.txt line format: {line!r}"
        requirements_floors[match.group(1).lower()] = match.group(2)

    shared = set(pyproject_floors) & set(requirements_floors)
    assert "streamlit" in shared, "expected streamlit in both files' floors"
    mismatched = {
        name: (pyproject_floors[name], requirements_floors[name])
        for name in shared
        if pyproject_floors[name] != requirements_floors[name]
    }
    assert not mismatched, (
        f"pyproject.toml vs requirements.txt floors differ: {mismatched}"
    )


def test_precommit_ruff_version_can_parse_this_repos_own_ruff_config() -> None:
    import re

    import yaml

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    (ruff_repo,) = [r for r in config["repos"] if "ruff-pre-commit" in r["repo"]]
    match = re.match(r"^v(\d+)\.(\d+)\.(\d+)$", ruff_repo["rev"])
    assert match, f"unexpected rev format: {ruff_repo['rev']!r}"
    assert (int(match.group(1)), int(match.group(2))) >= (0, 12), (
        f"ruff-pre-commit rev {ruff_repo['rev']} predates 0.12.0, the first "
        "version that can even parse this repo's own UP047 ignore entry"
    )


def test_precommit_mypy_uses_project_environment_and_cis_scope() -> None:
    import yaml

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    (local_repo,) = [r for r in config["repos"] if r["repo"] == "local"]
    (mypy_hook,) = [h for h in local_repo["hooks"] if h["id"] == "mypy"]
    assert mypy_hook["entry"] == "python -m mypy"
    assert mypy_hook["language"] == "system"
    assert "additional_dependencies" not in mypy_hook

    assert mypy_hook.get("pass_filenames") is False, (
        "mypy hook must set pass_filenames: false -- it cannot correctly "
        "type-check a subset of changed files in isolation"
    )
    assert mypy_hook.get("always_run") is True
    assert set(mypy_hook.get("args", [])) >= {"src", "tests", "scripts"}, (
        "mypy hook's args must cover the same scope as CI's own "
        "`mypy src tests scripts`"
    )

    ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--extra dev --extra dashboard --extra yahoo --extra docs" in ci
    assert "--extra notebooks" in ci


def test_streamlit_usage_telemetry_is_disabled_locally_and_in_docker() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[2]
    streamlit_config = tomllib.loads(
        (root / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    )
    assert streamlit_config["browser"]["gatherUsageStats"] is False
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false" in dockerfile


def test_slice_range_drops_a_look_ahead_hourly_bar() -> None:
    from quantlab.data.loader import DataLoader

    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-05 22:30", "2024-01-05 23:30"]),
            "symbol": "BTCUSDT",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adjusted_close": 1.0,
            "volume": 100.0,
        }
    )

    narrow = DataLoader._slice_range(
        data, date(2024, 1, 5), date(2024, 1, 5), "1h", is_247_market=True
    )
    # The 22:30 bar's bucket (23:30) closes before the request's own end
    # boundary; the 23:30 bar's bucket (00:30 the next day) does not.
    assert list(narrow["timestamp"]) == [pd.Timestamp("2024-01-05 22:30")]

    wider = DataLoader._slice_range(
        data, date(2024, 1, 5), date(2024, 1, 6), "1h", is_247_market=True
    )
    assert len(wider) == 2


def test_trading_day_helpers_normalize_the_time_component() -> None:
    """Both helpers must return a midnight-normalized date, discarding any
    time-of-day component on the input, so downstream comparisons never
    depend on what time within the day the input carried."""
    from quantlab.data.calendar import (
        first_trading_day_on_or_after,
        last_trading_day_on_or_before,
    )

    assert last_trading_day_on_or_before(
        pd.Timestamp("2024-01-06 15:30"), is_247_market=False
    ) == pd.Timestamp("2024-01-05")
    assert first_trading_day_on_or_after(
        pd.Timestamp("2024-01-07 15:30"), is_247_market=False
    ) == pd.Timestamp("2024-01-08")


def test_config_rejects_nan_and_infinity_in_bare_float_fields() -> None:
    from quantlab.config import BacktestConfig

    with pytest.raises(ValidationError, match="NaN/Infinity"):
        BacktestConfig(risk_free_rate=float("nan"))
    with pytest.raises(ValidationError, match="NaN/Infinity"):
        BacktestConfig(risk_free_rate=float("inf"))


def test_config_rejects_infinity_even_with_a_gt_bound() -> None:
    from quantlab.config import BacktestConfig, ExecutionConfig, PortfolioConfig

    with pytest.raises(ValidationError, match="NaN/Infinity"):
        BacktestConfig(initial_capital=float("inf"))
    with pytest.raises(ValidationError, match="NaN/Infinity"):
        ExecutionConfig(commission_bps=float("inf"))
    with pytest.raises(ValidationError, match="NaN/Infinity"):
        PortfolioConfig(maximum_leverage=float("inf"))


def test_config_rejects_nan_in_numpy_floats_and_sets() -> None:
    import numpy as np

    from quantlab.config import StrategyConfig

    for bad_threshold in (np.float32("nan"), np.float16("inf"), np.float32(0.5)):
        with pytest.raises(ValidationError, match="numpy scalar"):
            StrategyConfig(name="x", parameters={"threshold": bad_threshold})
    for bad_values in ({1.0, float("nan")}, {1.0, 2.0}):
        with pytest.raises(ValidationError, match="set"):
            StrategyConfig(name="x", parameters={"values": bad_values})
    # A plain (non-numpy) float NaN/Infinity is still rejected with the
    # NaN/Infinity message specifically, not the generic JSON-safe one.
    with pytest.raises(ValidationError, match="NaN/Infinity"):
        StrategyConfig(name="x", parameters={"threshold": float("nan")})


def test_config_rejects_nan_through_a_full_nested_experiment_config() -> None:
    cfg_dict = ExperimentConfig.from_yaml(
        Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    ).model_dump(mode="json")
    cfg_dict["backtest"]["initial_capital"] = float("inf")
    with pytest.raises(ValidationError, match="NaN/Infinity"):
        ExperimentConfig(**cfg_dict)
