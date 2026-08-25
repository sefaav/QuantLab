"""Integration test for report generation and saving."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tests.conftest import geometric_series, make_ohlcv

from quantlab.backtesting.result import BacktestResult
from quantlab.backtesting.runner import run_backtest_from_config
from quantlab.config import ExperimentConfig


def _result(tmp_seed: int = 1) -> BacktestResult:
    frames = []
    for sym, seed in [("SPY", 1), ("QQQ", 2), ("TLT", 3)]:
        prices = geometric_series(500, mu=0.0005, sigma=0.012, s0=100.0, seed=seed)
        frames.append(make_ohlcv(sym, prices, start="2019-01-01"))
    data = pd.concat(frames, ignore_index=True)
    cfg = ExperimentConfig.from_dict(
        {
            "experiment_name": "report_test",
            "data": {
                "instruments": [
                    {"symbol": s, "source": "csv", "calendar": "XNYS"}
                    for s in ["SPY", "QQQ", "TLT"]
                ],
                "start_date": "2019-01-01",
                "end_date": "2020-12-01",
            },
            "strategy": {
                "name": "cross_sectional_momentum",
                "parameters": {
                    "lookback_period": 100,
                    "skip_period": 5,
                    "top_fraction": 0.4,
                },
            },
            "portfolio": {"allocator": "inverse_volatility", "maximum_weight": 0.6},
            "execution": {
                "commission_bps": 2.0,
                "spread_bps": 3.0,
                "slippage_bps": 2.0,
            },
            "backtest": {
                "initial_capital": 100_000,
                "benchmark": {"symbol": "SPY", "source": "csv", "calendar": "XNYS"},
            },
        }
    )
    return run_backtest_from_config(data, cfg)


def test_html_report_contains_key_sections() -> None:
    html = _result().to_html()
    for heading in [
        "Executive summary",
        "Research question",
        "Hypothesis",
        "Methodology",
        "Results",
        "Limitations",
        "Conclusion",
    ]:
        assert heading in html
    # Honest-language guardrails: forbidden over-claims must be absent.
    for banned in ["guaranteed to work", "beaten the market", "will be profitable"]:
        assert banned not in html.lower()
    # Disclaimer present.
    assert "not investment advice" in html.lower()


def test_save_writes_full_artifact_set(tmp_path: Path) -> None:
    out = _result().save(tmp_path / "exp")
    expected = {
        "config.yaml",
        "metadata.json",
        "metrics.json",
        "equity_curve.csv",
        "trades.csv",
        "positions.csv",
        "costs.csv",
        "report.html",
    }
    present = {p.name for p in out.iterdir()}
    assert expected <= present
    assert (out / "figures").is_dir()
    # metrics.json is valid JSON with headline keys.
    metrics = json.loads((out / "metrics.json").read_text())
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics


def test_saved_config_roundtrips(tmp_path: Path) -> None:
    result = _result()
    out = result.save(tmp_path / "exp")
    reloaded = ExperimentConfig.from_yaml(out / "config.yaml")
    assert reloaded.experiment_name == result.config.experiment_name
