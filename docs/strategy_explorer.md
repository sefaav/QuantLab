# Strategy Explorer

The dashboard's **Strategies** mode is a research/education surface, separate
from Backtest/Walk-forward: a gallery of the built-in strategies, each
opening a detail page that explains how it works and lets you interact with
its own dedicated laboratory, picked via each lab's own "Data source"
control (see `shared_components.render_symbol_and_source_picker`): a
downloaded Yahoo or Binance symbol, a local CSV file, or QuantLab's
bundled, offline, synthetic demo dataset (not real historical prices,
used by default so a lab keeps working with zero setup). Switch source or
symbol, or turn off "Allow bundled synthetic demo data" once real local
files exist under `data/raw`, to research an actual instrument instead.

The laboratories reuse QuantLab's `DataLoader` and `ExperimentConfig`
machinery, but construct a simplified experiment where every selected
symbol shares ONE calendar (unlike the main dashboard's own per-instrument
calendar table). A Yahoo symbol's calendar is auto-detected from its
suffix where possible (e.g. `.HK` -> `XHKG`) and editable; a CSV symbol
carries no such signal and defaults to `XNYS`, also editable. Selecting
symbols that would need more than one calendar is rejected with a clear
error, since a lab computes on a single shared price matrix and cannot
represent more than one calendar at once (see
[Limitations](limitations.md)).

Launch it with `quantlab dashboard`, then switch the top **Mode** selector to
**Strategies**. The sidebar is hidden in this mode — everything happens on
the full-width gallery/detail pages instead.

## What a strategy's page shows

Each registered strategy gets a title and up to ten collapsible sections:
Overview, Economic intuition, Mathematical definition & signals,
Assumptions, Diagnostics, Parameters, Interactive laboratory, Interpretation,
Limitations & failure modes, and an optional References / Further reading
section at the end. The **Parameters** section documents every constructor
parameter the strategy accepts — including `price_type`/`periods_per_year`
where the runner injects them structurally — with what it is, where in the
signal pipeline it acts, why it exists, its default/typical range, the
effect of increasing/decreasing it, its trade-offs, and its interactions
with other parameters.

The **Interactive laboratory** is strategy-specific, not a generic template:
it lets you pick real data, adjust several of the strategy's own parameters
with widgets, and immediately recomputes and re-plots the relevant
indicator/signal/spread/threshold — Streamlit's rerun-on-interaction model
is what makes a parameter's effect observable without a bespoke "impact"
widget (see `quantlab.dashboard.explorer.shared_components.
render_price_chart`). Coverage varies by strategy and by parameter -- not
every constructor parameter necessarily has its own interactive control
yet; the Parameters section's text is the documented reference regardless.
Every lab loads its price data through one shared, bounded cache
(`shared_components.load_explorer_prices_cached`, `max_entries=32`), and
each strategy's Interactive laboratory expander only actually runs its body
once opened (a stateful/lazy `st.expander`, not a plain one) -- visiting the
page, or interacting with any OTHER widget on it, does not re-trigger the
lab's own computation.

## Architecture

Two parallel registries mirror each other:

- `quantlab.strategies.base` — the trading logic itself (`available_strategies()`,
  `strategy_parameter_names()`).
- `quantlab.dashboard.explorer.profile` — the *pedagogical content* for each
  strategy (`available_profiles()`, `get_profile()`), kept deliberately
  separate so UI/content and trading logic never mix.

The Strategy Explorer's own dispatch — the gallery/detail pages and the
optional Results-tab/report diagnostics described below — is profile-driven
and never names a specific strategy: it is built entirely from
`available_strategies()` + `get_profile()`, so a strategy without a
registered profile still gets a gallery card, with a "documentation coming
soon" placeholder, instead of silently disappearing. This does NOT mean
`app.py`/`cli.py` are strategy-name-free everywhere: the regular Backtest/
Walk-forward sidebar still branches on a strategy's name to render its own
config widgets (unrelated to the Strategy Explorer); `reporting/
html_report.py` is the one file that genuinely never names a strategy
(enforced by a dedicated regression test). Each strategy's own file under
`dashboard/explorer/profiles/` registers a `StrategyProfile` (markdown
fields, a `ParameterDoc` per parameter, and a `lab` callback) at import
time, the same pattern `register_strategy` already uses for trading logic
(see [Strategies](strategies.md#adding-a-new-strategy)).

### Results-tab / report diagnostics (optional, per strategy)

A strategy can optionally declare extra diagnostics that show up in the
regular Backtest Results tab and in the generated HTML report — every
built-in strategy except `buy_and_hold` currently does (its signal is
price availability alone, with no indicator/spread/ranking of its own to
diagnose). `pairs_trading` is the richest example, surfacing correlation/
hedge-ratio/spread/cointegration diagnostics under a "Pair relationship
diagnostics" section; the others surface a diagnostic suited to their own
signal (e.g. `mean_reversion`'s centered indicator and state-signal
charts). This is declared once, per strategy, on its own profile, via
`StrategyProfile.results_diagnostics`:

```python
ResultsDiagnostics(
    key="pair_diagnostics",  # robustness-dict / session_state key
    compute=...,  # (price_frame, ExperimentConfig) -> structured result
    render=...,  # (st, structured_result) -> None, for the Results tab
    report_section=...,  # structured_result -> DiagnosticsSection, for the HTML report
)
```

`app.py` (Backtest mode's Results tab and downloaded report, not
Walk-forward, see below) and `cli.py` (`backtest`/`report`, not
`walk-forward`) each just ask "does the current strategy's profile declare
`results_diagnostics`" and, if so, call `compute`/`render`/`report_section`
generically — neither contains a strategy name, and the HTML report
renders it under its own "Strategy diagnostics" heading, kept structurally
separate from "Robustness" (a correlation/spread/ADF diagnostic describes
whether the strategy's own assumptions hold, not whether the backtest
result is robust to cost/parameter/regime perturbation). A strategy
without `results_diagnostics` (the default) renders nothing — never an
empty section. If a diagnostic's own computation raises, that failure is
surfaced as a visible warning (Results tab and report) rather than the
section silently vanishing.

**Not shown for Walk-forward**: each fold can select different strategy
parameters than the base config and covers only that fold's own slice of
history, so a diagnostic computed once, on the full history with the base
config's parameters, would not accurately describe what any individual
fold actually traded. Walk-forward mode shows an explanatory caption
instead, for a strategy whose profile declares `results_diagnostics`.

## Reusable analytics layer

The statistics themselves live under `quantlab.features.*`, independent of
Streamlit, and are reused identically by the dashboard, the HTML report, and
(for pairs trading) `notebooks/04_pairs_trading_research.ipynb`:

- `features/stationarity.py` — `adf_test()`, `cointegration_test()`,
  `hurst_exponent()`, returning structured `ADFResult`/`CointegrationResult`
  dataclasses (statistic/p-value/critical values/H0-H1 verdict/plain-language
  interpretation), never a bare float.
- `features/correlation.py` — `correlation_matrix()`, used directly by the
  Pairs Trading lab's universe-screening step (not by the Results tab or
  report, which instead get a single pair's `correlation`/
  `rolling_correlation` from `PairDiagnostics` below).
- `features/pairs_diagnostics.py` — `compute_pair_diagnostics()` centralizes
  the pair diagnostics used by the lab, the Results tab and the report: a
  pair's hedge ratio, spread, configured spread indicator (zscore/rsi/
  percentile via `indicator`, defaulting to zscore but not always it -- see
  `PairDiagnostics.indicator`), half-life, and time-stability diagnostics.
  Its `adf_result`/`cointegration_result` are exploratory (one test over
  the whole sample). Its `rolling_adf_pvalue` calls the same underlying
  `periodic_stationarity_pvalues()` function the live strategy's entry gate
  uses (never a separately-computed approximation), but the diagnostic and
  the live strategy can receive different timelines in a mixed-calendar
  experiment (the live gate evaluates on the intersection of both legs'
  native session dates; this diagnostic uses the full combined timeline),
  so their results are not guaranteed to match exactly -- see
  [Limitations](limitations.md).

`dashboard/explorer/shared_components.py` holds the UI building blocks on
top of these (price/indicator charts, a correlation heatmap, a stationarity-
result card) — presentation only, no calculation, reused across every lab
and the Results-tab component in `dashboard/components.py`.

## Adding Strategy Explorer support for a new strategy

1. Register the strategy itself first (see
   [Strategies](strategies.md#adding-a-new-strategy)).
2. Add `dashboard/explorer/profiles/<name>.py`: write the markdown fields, a
   `ParameterDoc` for every entry in `strategy_parameter_names("<name>")`
   (including any structurally-injected ones your strategy reads), a `lab`
   callback, and call `register_profile(StrategyProfile(...))` at module
   level.
3. Add `dashboard/explorer/labs/<name>.py`: an interactive `render(st)`
   loading data through the shared, bounded cache
   (`shared_components.load_explorer_prices_cached`) and the strategy's own
   `quantlab.features.*` functions — never a second implementation of the
   strategy's math. Prefer calling the strategy class itself
   (`quantlab.strategies.<name>.<Name>Strategy(...).generate_signals(data)`)
   over reimplementing its state machine when a lab needs the strategy's
   real signal (its own entry/exit/stop state machine's output -- still
   upstream of the allocator, portfolio constraints, rebalancing schedule,
   execution delay and accounting that together determine the actual
   executed position), not just an intermediate indicator (see the Mean
   Reversion lab's "State machine" section for an example).
4. Register the profile module in `dashboard/explorer/profiles/__init__.py`.
5. Only if the strategy needs its own Results-tab/report section: declare
   `results_diagnostics` on the profile, following the pairs_trading example
   above.
6. Add it to `tests/unit/test_dashboard_explorer_profiles.py`'s coverage
   (parametrized over `available_strategies()`, so a new strategy is picked
   up automatically) and, if it declares `results_diagnostics`, a Results-tab
   assertion alongside the existing pairs_trading one in `test_dashboard.py`.

For the Results-tab/report diagnostics mechanism specifically, no change to
`app.py`, `cli.py`, or `reporting/html_report.py` is needed for a strategy
that doesn't declare `results_diagnostics` (that machinery is entirely
generic). This does not extend to the rest of `app.py`: a genuinely new
strategy still needs its own sidebar config widgets added to the regular
Backtest/Walk-forward mode, alongside its Strategy Explorer profile.
