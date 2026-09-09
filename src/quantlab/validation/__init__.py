"""Out-of-sample validation and robustness tools."""

from __future__ import annotations

from quantlab.validation.bootstrap import BootstrapResult, bootstrap_returns
from quantlab.validation.holdout import (
    HoldoutReport,
    compute_holdout_split,
    run_holdout_report,
    run_holdout_validation,
)
from quantlab.validation.panel_inference import (
    ClusterDifferenceResult,
    ClusterSpearmanResult,
    FamaMacBethResult,
    PairedSharpeDifferenceResult,
    PermutationTestResult,
    Statistic,
    cliffs_delta,
    cluster_bootstrap_difference,
    clustered_regression,
    fama_macbeth_difference,
    holm_adjust,
    paired_cluster_bootstrap_sharpe_difference,
    per_cluster_spearman,
    within_cluster_permutation_test,
)
from quantlab.validation.parameter_grid import (
    default_parameter_grid,
    parameter_grid_for_config,
)
from quantlab.validation.parameter_sensitivity import (
    run_parameter_sensitivity,
    run_walk_forward_parameter_sensitivity,
    sensitivity_heatmap_data,
)
from quantlab.validation.robustness import (
    monte_carlo_permutation,
    run_stress_tests,
    run_walk_forward_stress_tests,
    stress_test_checkpoint_paths,
)
from quantlab.validation.splits import (
    ChronologicalSplit,
    FormationTestWindow,
    WalkForwardWindow,
    chronological_split,
    formation_test_windows,
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
    "ClusterDifferenceResult",
    "ClusterSpearmanResult",
    "FamaMacBethResult",
    "FoldResult",
    "FormationTestWindow",
    "HoldoutReport",
    "PairedSharpeDifferenceResult",
    "PermutationTestResult",
    "Statistic",
    "WalkForwardResult",
    "WalkForwardValidator",
    "WalkForwardWindow",
    "bootstrap_returns",
    "chronological_split",
    "cliffs_delta",
    "cluster_bootstrap_difference",
    "clustered_regression",
    "compute_holdout_split",
    "default_parameter_grid",
    "fama_macbeth_difference",
    "formation_test_windows",
    "holm_adjust",
    "monte_carlo_permutation",
    "paired_cluster_bootstrap_sharpe_difference",
    "parameter_grid_for_config",
    "per_cluster_spearman",
    "run_holdout_report",
    "run_holdout_validation",
    "run_parameter_sensitivity",
    "run_stress_tests",
    "run_walk_forward_parameter_sensitivity",
    "run_walk_forward_stress_tests",
    "sensitivity_heatmap_data",
    "stress_test_checkpoint_paths",
    "walk_forward_windows",
    "within_cluster_permutation_test",
]
