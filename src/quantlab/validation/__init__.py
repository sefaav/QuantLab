"""Out-of-sample validation and robustness tools."""

from __future__ import annotations

from quantlab.validation.bootstrap import BootstrapResult, bootstrap_returns
from quantlab.validation.holdout import (
    HoldoutReport,
    compute_holdout_split,
    run_holdout_report,
    run_holdout_validation,
)
from quantlab.validation.parameter_grid import (
    default_parameter_grid,
    parameter_grid_for_config,
)
from quantlab.validation.parameter_sensitivity import (
    run_parameter_sensitivity,
    sensitivity_heatmap_data,
)
from quantlab.validation.robustness import (
    monte_carlo_permutation,
    run_stress_tests,
)
from quantlab.validation.splits import (
    ChronologicalSplit,
    WalkForwardWindow,
    chronological_split,
    walk_forward_windows,
)
from quantlab.validation.walk_forward import (
    FoldResult,
    WalkForwardResult,
    WalkForwardValidator,
)

__all__ = [
    "BootstrapResult",
    "ChronologicalSplit",
    "FoldResult",
    "HoldoutReport",
    "WalkForwardResult",
    "WalkForwardValidator",
    "WalkForwardWindow",
    "bootstrap_returns",
    "chronological_split",
    "compute_holdout_split",
    "default_parameter_grid",
    "monte_carlo_permutation",
    "parameter_grid_for_config",
    "run_holdout_report",
    "run_holdout_validation",
    "run_parameter_sensitivity",
    "run_stress_tests",
    "sensitivity_heatmap_data",
    "walk_forward_windows",
]
