# QuantLab

**Reproducible quantitative research and backtesting.**

QuantLab turns a financial hypothesis into a reproducible, bias-aware experiment:
download data, clean and validate it, build features and signals, run a
vectorised backtest — with a delayed-execution barrier that prevents common
look-ahead leakage — with realistic costs, measure performance and risk,
validate out-of-sample, and generate an honest research report — all driven
by one YAML config. Custom strategies remain responsible for their own causal
feature and signal construction.

> This project is for **educational and research purposes only**. It is not
> investment advice, and historical performance does not guarantee future
> results.

## Where to start

- [Architecture](architecture.md) — how the modules fit together and why the
  pipeline is ordered the way it is.
- [Data pipeline](data_pipeline.md) — sources, canonical schema, cleaning,
  validation, storage.
- [Strategies](strategies.md) — the strategy contract and how to add a new one.
- [Backtesting](backtesting.md) — the accounting model and look-ahead-bias
  prevention.
- [Validation](validation.md) — walk-forward, sensitivity, bootstrap, stress
  tests.
- [Limitations](limitations.md) — what this platform does *not* model, stated
  plainly.

## Quick start

```bash
pip install -e ".[dev,dashboard,yahoo,extra]"
quantlab backtest --config configs/momentum_sp500.yaml
quantlab dashboard
```

Run `quantlab --help` for the complete command list and
`quantlab <command> --help` for command-specific options.
