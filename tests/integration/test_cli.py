"""Integration tests for the Typer CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tests.conftest import geometric_series, make_ohlcv
from typer.testing import CliRunner

from quantlab.cli import app

runner = CliRunner()


def _write_offline_experiment(tmp_path: Path) -> tuple[Path, Path]:
    """Create CSV data + a config using the csv source under tmp_path."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for sym, seed in [("AAA", 1), ("BBB", 2), ("CCC", 3)]:
        prices = geometric_series(500, mu=0.0005, sigma=0.012, s0=100.0, seed=seed)
        make_ohlcv(sym, prices, start="2019-01-01").to_csv(
            raw / f"{sym}.csv", index=False
        )
    config = {
        "experiment_name": "cli_test",
        "data": {
            "source": "csv",
            "symbols": ["AAA", "BBB", "CCC"],
            "start_date": "2019-01-01",
            "end_date": "2020-12-31",
            "market_calendar": "XNYS",
        },
        "strategy": {
            "name": "cross_sectional_momentum",
            "parameters": {
                "lookback_period": 100,
                "skip_period": 5,
                "top_fraction": 0.5,
            },
        },
        "portfolio": {"allocator": "inverse_volatility", "maximum_weight": 0.6},
        "execution": {"commission_bps": 2.0, "spread_bps": 3.0, "slippage_bps": 2.0},
        "backtest": {"initial_capital": 100000, "benchmark_symbol": "AAA"},
    }
    config_path = tmp_path / "cli_test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, raw


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["download", "backtest", "walk-forward", "report", "dashboard"]:
        assert command in result.stdout


def test_cli_backtest_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, raw = _write_offline_experiment(tmp_path)
    # Point the DataLoader's default raw dir at our temp CSVs.
    import quantlab.data.loader as loader_mod

    monkeypatch.setattr(loader_mod, "RAW_DATA_DIR", raw)

    out = tmp_path / "out"
    result = runner.invoke(
        app, ["backtest", "--config", str(config_path), "--output", str(out)]
    )
    assert result.exit_code == 0, result.stdout
    assert (out / "report.html").is_file()
    assert (out / "metrics.json").is_file()


def test_cli_backtest_bad_config_exits_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: a: valid: config", encoding="utf-8")
    result = runner.invoke(app, ["backtest", "--config", str(bad)])
    assert result.exit_code != 0


def test_cli_missing_config_exits_nonzero() -> None:
    result = runner.invoke(app, ["backtest", "--config", "does_not_exist.yaml"])
    assert result.exit_code != 0
