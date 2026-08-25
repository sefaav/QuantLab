"""Generate factual, deliberately cautious research-report text."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantlab.backtesting.result import BacktestResult


STANDARD_LIMITATIONS: list[str] = [
    "Adjusted-price series may incorporate dividends, splits and other corporate "
    "actions more cleanly than they could have been traded in practice.",
    "A fixed symbol list does not reconstruct point-in-time universe membership, "
    "so selection or survivorship bias may be present.",
    "Execution models approximate trading costs and do not reproduce an order book.",
    "No hard fill-probability or market-capacity constraint is modelled.",
    "No regulatory constraints, short-borrow limits or margin rules are applied.",
    "No taxes are modelled.",
    "Parameter and asset choices carry a risk of data-snooping.",
    "Historical results are not predictive of future performance.",
]


def _finite_number(value: object) -> float | None:
    """Return a finite real number, excluding booleans."""
    if (
        value is None
        or isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _format_metric(value: object, kind: str) -> str:
    number = _finite_number(value)
    if number is None:
        return "n/a"
    if kind == "pct":
        return f"{number:.2%}"
    if kind == "currency":
        return f"{number:,.2f}"
    return f"{number:.2f}"


def _actually_used(result: BacktestResult, key: str, fallback: Any) -> Any:
    """Prefer the object actually executed over ``result.config``'s own value.

    ``BacktestEngine`` can be used directly with a custom strategy,
    allocator or execution-model instance that need not match
    ``config``'s own YAML-derived settings (see docs/api.md's "Extension
    points") -- ``result.metadata``'s ``strategy``/``allocator``/
    ``commission_bps``/``spread_bps`` fields record what was actually run,
    so report text must read from there, not from ``result.config``, or it
    could describe a different reality than the one accounting and the
    trade log actually charged. Falls back to ``result.config`` only for an
    older saved result or a duck-typed stand-in missing the field.
    """
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get(key)
        if value is not None:
            return value
    return fallback


def _actual_period(result: BacktestResult) -> tuple[str, str] | None:
    """Return the dates actually represented by the result."""
    index = result.equity_curve.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        return None
    return index.min().date().isoformat(), index.max().date().isoformat()


def executive_summary(result: BacktestResult) -> str:
    """Return a one-paragraph summary of the realised backtest sample."""
    metrics = result.metrics
    cfg = result.config
    actual_period = _actual_period(result)
    period_text = (
        f"from {actual_period[0]} to {actual_period[1]}"
        if actual_period is not None
        else "over an unavailable observed period"
    )
    oos_scope = out_of_sample_scope(result)
    scope_text = (
        f" These are {oos_scope} results, not a full-sample fit."
        if oos_scope is not None
        else ""
    )

    benchmark_text = ""
    if result.benchmark_returns is not None:
        benchmark_label = cfg.benchmark_label or "configured benchmark"
        benchmark_text = (
            f" Against {benchmark_label}, estimated beta was "
            f"{_format_metric(metrics.get('beta'), 'num')} and approximate "
            f"annualised alpha was {_format_metric(metrics.get('alpha'), 'pct')}."
        )

    strategy_name = _actually_used(result, "strategy", cfg.strategy_name)
    return (
        f"The {strategy_name.replace('_', ' ')} strategy was tested on "
        f"{len(cfg.symbols)} instrument(s) {period_text}.{scope_text} Net of modelled "
        f"transaction costs, total return was "
        f"{_format_metric(metrics.get('total_return'), 'pct')} "
        f"(CAGR {_format_metric(metrics.get('cagr'), 'pct')}), annualised volatility "
        f"was {_format_metric(metrics.get('annualized_volatility'), 'pct')}, "
        f"Sharpe was {_format_metric(metrics.get('sharpe_ratio'), 'num')} and maximum "
        f"drawdown was {_format_metric(metrics.get('max_drawdown'), 'pct')}. Total "
        f"modelled trading cost was {_format_metric(result.total_costs(), 'currency')} "
        f"currency units.{benchmark_text} These figures are historical and conditional "
        "on the tested assumptions."
    )


def research_question(result: BacktestResult) -> str:
    """Return a question that names only configured and attached evidence."""
    cfg = result.config
    strategy = _actually_used(result, "strategy", cfg.strategy_name)
    portfolio = cfg.portfolio
    volatility_targeted = (
        portfolio.allocator == "volatility_targeting"
        or portfolio.target_volatility is not None
    )
    sample = "out-of-sample" if _oos_metrics(result) is not None else "historical"
    if "cross_sectional_momentum" in strategy:
        direction = (
            "long/short" if cfg.strategy_parameters.get("long_short") else "long-only"
        )
        question = (
            f"Does {direction} cross-sectional momentum produce robust "
            f"{sample} risk-adjusted returns across the tested universe "
            "after modelled transaction costs"
        )
    elif "time_series_momentum" in strategy:
        question = (
            f"Does trailing time-series momentum produce robust {sample} "
            "returns after modelled transaction costs"
        )
    elif "mean_reversion" in strategy:
        question = (
            f"Does price mean reversion produce robust {sample} "
            "returns after modelled transaction costs"
        )
    elif "pairs" in strategy:
        question = (
            f"Does an ADF-filtered pairs spread produce profitable {sample} "
            "reversion after modelled transaction costs"
        )
    elif "trend" in strategy:
        question = (
            f"Does trend following produce robust {sample} risk-adjusted "
            "returns after modelled transaction costs"
        )
    else:
        question = (
            f"Does the {strategy.replace('_', ' ')} strategy produce robust "
            f"{sample} risk-adjusted returns after modelled costs"
        )
    suffix = (
        ", under volatility-targeted position sizing?" if volatility_targeted else "?"
    )
    return question + suffix


def hypothesis(result: BacktestResult) -> str:
    """Return H1/H0 and state whether OOS evidence is attached to the run."""
    oos = _oos_metrics(result)
    oos_status = (
        f"attached to this run: {oos[1]}."
        if oos is not None
        else "not attached to this run — the metrics above are full-sample only."
    )
    return (
        "H1: net of modelled transaction costs, the strategy has a genuine "
        "positive risk-adjusted edge (Sharpe > 0). H0: any apparent edge is "
        "indistinguishable from noise or is eliminated by trading costs. "
        "Validating H1 needs both out-of-sample evidence (holdout or "
        "walk-forward) and stress-test/robustness evidence — see the "
        "Robustness section for what this report attaches. Out-of-sample "
        f"evidence is {oos_status}"
    )


def _oos_metrics(result: BacktestResult) -> tuple[dict[str, Any], str] | None:
    """Return attached OOS metrics, preferring walk-forward evidence.

    Some callers (table builders in particular) are exercised with minimal
    duck-typed stand-ins that don't implement the full ``BacktestResult``
    interface, so a missing ``metadata`` attribute is treated the same as
    an empty one rather than raising.
    """
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    candidates = (
        (
            "walk_forward_oos_metrics",
            "out-of-sample (walk-forward test folds only)",
        ),
        (
            "holdout_chronological_metrics",
            "chronological holdout test block (out-of-sample only if "
            "strategy/parameter choices were frozen before it was inspected)",
        ),
    )
    for key, label in candidates:
        metrics = metadata.get(key)
        if isinstance(metrics, Mapping) and metrics:
            return dict(metrics), label
    return None


def out_of_sample_scope(result: BacktestResult) -> str | None:
    """Return the OOS scope label only when ``result.metrics`` *is* that series.

    Not just whenever some OOS evidence is attached: a holdout result also
    carries chronological-test-block evidence, but its own ``metrics`` stay
    a genuine full-sample fit (the holdout block's metrics are held
    separately, in ``metadata["holdout_chronological_metrics"]``) —
    "full-sample" is still correct there. Only a walk-forward result's
    ``metrics`` already are the out-of-sample series, so only that case
    needs "full-sample" labels corrected elsewhere (e.g. Results headings,
    ``tables.subperiod_table``'s aggregate row).
    """
    oos = _oos_metrics(result)
    if oos is None:
        return None
    oos_metrics, scope = oos
    return scope if oos_metrics == result.metrics else None


def _portfolio_methodology(result: BacktestResult) -> str:
    portfolio = result.config.portfolio
    allocator_name = _actually_used(result, "allocator", portfolio.allocator)
    details = [
        f"allocator {allocator_name}",
        f"rebalance cadence {portfolio.rebalance_frequency}",
        f"maximum leverage {portfolio.maximum_leverage:.2f}x",
    ]
    if portfolio.long_only:
        details.append("long-only positions")
    if portfolio.target_volatility is not None:
        details.append(
            f"annual volatility target {portfolio.target_volatility:.2%} "
            f"over {portfolio.volatility_window} observations"
        )
    optional = (
        (portfolio.maximum_weight, "maximum absolute weight", ".2%"),
        (portfolio.target_minimum_weight, "target minimum weight", ".2%"),
        (portfolio.maximum_gross_exposure, "maximum gross exposure", ".2f"),
        (portfolio.maximum_net_exposure, "maximum absolute net exposure", ".2f"),
        (portfolio.maximum_turnover, "maximum L1 turnover per rebalance", ".2f"),
    )
    for value, label, spec in optional:
        if value is not None:
            details.append(f"{label} {format(value, spec)}")
    if portfolio.target_maximum_positions is not None:
        details.append(f"target maximum positions {portfolio.target_maximum_positions}")
    return ", ".join(details)


def _execution_methodology(result: BacktestResult) -> str:
    execution = result.config.execution
    commission_bps = float(
        _actually_used(result, "commission_bps", execution.commission_bps)
    )
    spread_bps = float(_actually_used(result, "spread_bps", execution.spread_bps))
    model = str(
        _actually_used(result, "slippage_model", execution.slippage_model)
    ).lower()
    slippage_bps = float(_actually_used(result, "slippage_bps", execution.slippage_bps))
    base = (
        f"commission {commission_bps:.1f} bps of traded notional and "
        f"a {spread_bps:.1f} bps full quoted spread (half charged when "
        "crossing)"
    )
    if model in {"volume", "volume_based"}:
        impact_coefficient = float(
            _actually_used(result, "impact_coefficient", execution.impact_coefficient)
        )
        slippage = (
            f"volume-based slippage with {slippage_bps:.1f} bps base "
            f"slippage plus square-root market impact using trailing dollar ADV "
            f"and impact coefficient {impact_coefficient:.4f}"
        )
    else:
        slippage = f"constant slippage {slippage_bps:.1f} bps"
    return f"{base}, and {slippage}"


def methodology(result: BacktestResult) -> str:
    """Describe portfolio construction, execution and attached validation."""
    cfg = result.config
    oos = _oos_metrics(result)
    if oos is not None:
        validation_text = (
            f"Attached validation: {oos[1]}. See the Robustness section for details."
        )
    else:
        validation_text = (
            "This report contains full-sample statistics only; no out-of-sample "
            "evidence is attached. Run `quantlab walk-forward`, or configure a "
            "chronological holdout with a non-zero test ratio, before drawing "
            "out-of-sample conclusions."
        )
    strategy_name = _actually_used(result, "strategy", cfg.strategy_name)
    return (
        f"The {strategy_name} strategy generates signals. Portfolio construction "
        f"uses {_portfolio_methodology(result)}. Execution costs use "
        f"{_execution_methodology(result)}. Weights are shifted by one observation "
        f"before earning returns. {validation_text}"
    )


def _bundled_demo_data_used(result: BacktestResult) -> bool | None:
    """Whether the bundled synthetic CSV fallback actually triggered.

    ``None`` when unknown (no attached data-quality report to consult, e.g.
    an older saved result, or a duck-typed stand-in without a full
    ``metadata`` attribute -- see ``_oos_metrics``) -- distinct from
    ``False`` (known not to have been used), so callers can fall back to a
    hedged statement only in the genuinely-unknown case.
    """
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    data_quality = metadata.get("data_quality")
    if (
        not isinstance(data_quality, dict)
        or "bundled_demo_data_used" not in data_quality
    ):
        return None
    return bool(data_quality["bundled_demo_data_used"])


def data_description(result: BacktestResult) -> str:
    """Describe requested data and the sample actually tested."""
    cfg = result.config
    actual_period = _actual_period(result)
    actual_text = (
        f"Observed backtest period: {actual_period[0]} to {actual_period[1]}."
        if actual_period is not None
        else "Observed backtest period: unavailable."
    )
    demo_text = ""
    if cfg.data.use_bundled_demo_data:
        demo_used = _bundled_demo_data_used(result)
        if demo_used is True:
            demo_text = " Bundled synthetic CSV fallback was used for this run."
        elif demo_used is False:
            demo_text = (
                " Bundled synthetic CSV fallback was enabled but not needed for "
                "this run (local CSV files were found)."
            )
        else:
            demo_text = (
                " Bundled synthetic CSV fallback was enabled; the saved data "
                "artefacts must be consulted to determine whether it was used."
            )
    return (
        f"Source: {cfg.data_source}. Instruments ({len(cfg.symbols)}): "
        f"{', '.join(cfg.symbols)}. Frequency: {cfg.frequency}. Requested period: "
        f"{cfg.start_date} to {cfg.end_date}. {actual_text} Missing-value policy: "
        f"{cfg.data.missing_value_policy}.{demo_text}"
    )


def limitations(result: BacktestResult) -> list[str]:
    """Return limitations that apply to this configuration."""
    items = list(STANDARD_LIMITATIONS)
    model = str(result.config.execution.slippage_model).lower()
    if model in {"volume", "volume_based"}:
        items.append(
            "Volume-based slippage uses a simplified square-root impact model and "
            "trailing dollar ADV, not realised fills or order-book depth."
        )
    else:
        items.append(
            "Constant slippage does not vary with order size, liquidity or volatility."
        )
    from quantlab.config import DataSourceName

    if any(
        instrument.source is DataSourceName.BINANCE
        for instrument in result.config.data.instruments
    ):
        items.append("Crypto data uses one venue rather than a consolidated tape.")
    if result.config.data.use_bundled_demo_data:
        demo_used = _bundled_demo_data_used(result)
        if demo_used is True:
            items.append(
                "Synthetic bundled CSV data was used for this run; it is suitable "
                "for demonstrations, not empirical market claims."
            )
        elif demo_used is None:
            items.append(
                "Synthetic bundled CSV data may have been used when local CSV "
                "files were absent; it is suitable for demonstrations, not "
                "empirical market claims."
            )
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, Mapping) and "holdout_chronological_metrics" in metadata:
        items.append(
            "The holdout evidence attached to this report is a chronological "
            "test block: data held back from the fitted metrics by a "
            "mechanical time split. That alone does not confirm it is "
            "genuinely out-of-sample -- it does not by itself confirm that "
            "strategy or parameter choices were frozen before this block was "
            "ever inspected. Its out-of-sample status depends on that "
            "discipline having been followed upstream of this report."
        )
    calendars = {instrument.calendar for instrument in result.config.data.instruments}
    if len(calendars) > 1:
        items.append(
            "Instruments span more than one calendar: rolling-window features "
            "(momentum lookback, volatility window, ADV window, technical "
            "indicators) count raw periods, not real trading sessions per "
            "instrument, so a session-bound instrument's estimates are diluted "
            "by the flat, zero-return/zero-volume bars inserted on its verified "
            "closures to keep the combined timeline dense."
        )
    return items


def _disposition(sharpe: object) -> str:
    value = _finite_number(sharpe)
    if value is None:
        return "insufficient metrics for a risk-adjusted assessment"
    if value > 0:
        return "positive historical characteristics"
    return "weak or negative historical characteristics"


def conclusion(result: BacktestResult) -> str:
    """State full-sample and attached OOS evidence without conflating them."""
    full_sharpe = result.metrics.get("sharpe_ratio")
    full_drawdown = result.metrics.get("max_drawdown")
    epilogue = (
        " Future performance remains uncertain and depends materially on "
        "execution costs, market regime and parameter stability. These "
        "results are not a guarantee of future profitability and are not "
        "investment advice."
    )
    oos = _oos_metrics(result)
    if oos is not None:
        oos_metrics, scope = oos
        oos_sharpe = oos_metrics.get("sharpe_ratio")
        oos_drawdown = oos_metrics.get("max_drawdown")
        if oos_metrics == result.metrics:
            # A walk-forward OOS result's `metrics` *is* the stitched
            # out-of-sample series (see WalkForwardValidator._build_oos_result)
            # — there is no separate full-sample fit to report alongside it.
            return (
                f"Under the tested assumptions, the strategy displayed "
                f"{_disposition(oos_sharpe)} using {scope} data (Sharpe "
                f"{_format_metric(oos_sharpe, 'num')}, maximum drawdown "
                f"{_format_metric(oos_drawdown, 'pct')})." + epilogue
            )
        return (
            f"Under the tested assumptions, the strategy displayed "
            f"{_disposition(oos_sharpe)} using {scope} data (out-of-sample Sharpe "
            f"{_format_metric(oos_sharpe, 'num')}, out-of-sample maximum drawdown "
            f"{_format_metric(oos_drawdown, 'pct')}). The separate full-sample "
            f"results are Sharpe {_format_metric(full_sharpe, 'num')} and maximum "
            f"drawdown {_format_metric(full_drawdown, 'pct')}." + epilogue
        )
    return (
        f"Under the tested assumptions, the strategy displayed "
        f"{_disposition(full_sharpe)} using full-sample data; no out-of-sample "
        f"validation is attached (Sharpe {_format_metric(full_sharpe, 'num')}, "
        f"maximum drawdown {_format_metric(full_drawdown, 'pct')})." + epilogue
    )
