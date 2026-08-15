# Python API

QuantLab exposes its documented entry points through package `__all__` lists
and the `quantlab.config` module. Import from the location shown below rather
than from private modules or names beginning with an underscore.

| Package | Main public objects |
| --- | --- |
| `quantlab.config` | `ExperimentConfig` and its validated configuration models |
| `quantlab.data` | `DataLoader`, `DataValidator`, `DataCleaner`, `ParquetStorage`, `Universe`, schema and resampling helpers |
| `quantlab.features` | Return, momentum, volatility, mean-reversion and technical features; optional `FeaturePipeline` |
| `quantlab.strategies` | Built-in strategies, registry helpers and `BaseStrategy` |
| `quantlab.portfolio` | Allocators, constraints, rebalancing and volatility targeting |
| `quantlab.execution` | Commission, spread, slippage and aggregate execution models |
| `quantlab.backtesting` | `BacktestEngine`, `BacktestResult`, accounting, benchmark and trade-log helpers |
| `quantlab.risk` | Performance, drawdown, exposure, VaR/CVaR and stress helpers |
| `quantlab.validation` | Holdout, walk-forward (`FoldResult`, `WalkForwardResult`, `WalkForwardValidator`), sensitivity, bootstrap and stress validation |
| `quantlab.reporting` | Tables, charts and HTML-report generation |

## Recommended entry point

For a configuration-driven experiment, use the runner so the same construction
path is shared by Python, the CLI and the dashboard:

```python
from quantlab.backtesting import run_backtest_from_config
from quantlab.config import ExperimentConfig
from quantlab.data import DataLoader

config = ExperimentConfig.from_yaml("configs/demo_offline.yaml")
data, quality_report = DataLoader().load(config)
result = run_backtest_from_config(
    data,
    config,
    data_quality_report=quality_report,
)

print(result.summary())
```

Use `BacktestEngine` directly when supplying custom strategy, allocator or
execution-model instances. Its `data` argument must be a pandas `DataFrame` in
QuantLab's canonical long OHLCV schema: one row per `(timestamp, symbol)` and
the columns listed in `quantlab.constants.OHLCV_COLUMNS`. The engine rejects a
missing `timestamp`/`symbol` axis and any configured tradable symbol absent
from the frame before running the strategy. Deliberate engine failures raise
`BacktestError`; deeper schema, strategy and configuration checks use the more
specific exceptions in `quantlab.exceptions`.

`WalkForwardValidator.run()` has the same canonical-data expectation and
requires every symbol in `config.data.symbols` to be present, even when the
history is too short to form a fold. It accepts a caller-provided parameter
grid and returns a `WalkForwardResult` containing public `FoldResult` records.

## Extension points

- Subclass `BaseStrategy`, implement `generate_signals()`, validate constructor
  parameters, then register the strategy with `register_strategy()`.
- Subclass `PortfolioAllocator` and register it with `register_allocator()`.
- Implement `SlippageModel` for a custom slippage assumption.
- Use `FeaturePipeline` only when an explicit reusable feature transformation
  pipeline is useful; built-in strategies call feature functions directly.

See [Strategies](strategies.md), [Backtesting](backtesting.md) and
[Validation](validation.md) for the contracts and methodological assumptions.
