"""Tests for dependence-aware panel inference (quantlab.validation.panel_inference).

Covers null-recovery (no real effect -> a CI/p-value that fails to reject),
positive-effect detection, and parameter-preservation for every function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.validation.panel_inference import (
    cliffs_delta,
    cluster_bootstrap_difference,
    clustered_regression,
    fama_macbeth_difference,
    holm_adjust,
    paired_cluster_bootstrap_sharpe_difference,
    per_cluster_spearman,
    within_cluster_permutation_test,
)

# --------------------------------------------------------------------------- #
# cluster_bootstrap_difference / within_cluster_permutation_test: null
# recovery, and holm_adjust
# --------------------------------------------------------------------------- #


def test_cluster_bootstrap_and_permutation_recover_null() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=600)
    treated = rng.random(600) < 0.3
    cluster = np.repeat(np.arange(30), 20)
    boot = cluster_bootstrap_difference(y, treated, cluster, n_boot=300, seed=0)
    assert boot.ci_low < 0 < boot.ci_high
    perm = within_cluster_permutation_test(y, treated, cluster, n_perm=300, seed=0)
    assert perm.p_value > 0.05


def test_holm_adjustment_monotone() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_holm_adjust_empty_and_single() -> None:
    assert holm_adjust([]) == []
    assert holm_adjust([0.02]) == pytest.approx([0.02])


def test_holm_adjust_rejects_out_of_range_or_missing() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjust([0.5, 1.5])
    with pytest.raises(ValueError, match="missing"):
        holm_adjust([0.5, float("nan")])


# --------------------------------------------------------------------------- #
# cluster_bootstrap_difference / within_cluster_permutation_test: positive
# effect and edge cases
# --------------------------------------------------------------------------- #


def test_cluster_bootstrap_difference_detects_a_real_effect() -> None:
    rng = np.random.default_rng(2)
    n_clusters, per_cluster = 20, 30
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    treated = rng.random(n_clusters * per_cluster) < 0.5
    y = rng.normal(size=n_clusters * per_cluster) + treated * 1.5  # large, real effect
    boot = cluster_bootstrap_difference(y, treated, cluster, n_boot=500, seed=0)
    assert boot.ci_low > 0  # CI excludes zero, correctly
    assert boot.estimate == pytest.approx(1.5, abs=0.5)


def test_within_cluster_permutation_test_detects_a_real_effect() -> None:
    rng = np.random.default_rng(2)
    n_clusters, per_cluster = 20, 30
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    treated = rng.random(n_clusters * per_cluster) < 0.5
    y = rng.normal(size=n_clusters * per_cluster) + treated * 1.5
    perm = within_cluster_permutation_test(y, treated, cluster, n_perm=500, seed=0)
    assert perm.p_value < 0.05


def test_cluster_bootstrap_difference_ignores_a_cluster_with_no_real_data() -> None:
    """Regression test: a cluster whose EVERY row has a NaN outcome
    contributes nothing real to the statistic (``func`` is already
    NaN-aware), but must ALSO be excluded from ``n_clusters`` and from
    the resampling population itself -- otherwise it silently inflates
    the reported cluster/treated/control counts and wastes resampling
    draws on a cluster that can never contribute a real observation,
    biasing the effective resampling variance of every OTHER cluster.
    Adding such a cluster must change nothing about the actual result:
    same estimate, same (recovered) cluster/treated/control counts as
    the baseline without it."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    treated = rng.random(100) < 0.5
    cluster = np.repeat(np.arange(10), 10)
    baseline = cluster_bootstrap_difference(y, treated, cluster, n_boot=200, seed=0)

    y_with_nan_cluster = np.concatenate([y, np.full(5, np.nan)])
    treated_with_nan_cluster = np.concatenate([treated, np.array([True] * 5)])
    cluster_with_nan_cluster = np.concatenate([cluster, np.full(5, 99)])
    with_nan_cluster = cluster_bootstrap_difference(
        y_with_nan_cluster,
        treated_with_nan_cluster,
        cluster_with_nan_cluster,
        n_boot=200,
        seed=0,
    )

    assert with_nan_cluster.n_clusters == baseline.n_clusters
    assert with_nan_cluster.n_treated == baseline.n_treated
    assert with_nan_cluster.n_control == baseline.n_control
    assert with_nan_cluster.estimate == pytest.approx(baseline.estimate)


def test_cluster_bootstrap_difference_rejects_a_cluster_wiped_out_by_nan() -> None:
    """Mirrors `within_cluster_permutation_test`'s identical guard: the
    up-front `>=2 distinct clusters` floor runs BEFORE the complete-case
    filter, so a cluster that vanishes entirely from the NaN-y drop must
    be re-checked on the complete case, not silently left at 1."""
    y = np.array([1.0, 2.0, np.nan, np.nan])
    treated = np.array([True, False, True, False])
    cluster = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="at least 2 distinct clusters"):
        cluster_bootstrap_difference(y, treated, cluster, n_boot=50, seed=0)


def test_within_cluster_permutation_test_ignores_nan_outcome_rows_in_the_shuffle() -> (
    None
):
    """A NaN-outcome row must be dropped ENTIRELY before permuting, never
    left in the shuffle pool where it could still receive a shuffled
    treated/control label -- proven by checking that adding extra
    NaN-outcome rows (with their own random cluster/treatment labels) to
    an otherwise-identical panel, under the SAME seed, produces IDENTICAL
    results. Leaving the extra rows in the shuffle pool would change each
    cluster's own array size and thus the RNG draw sequence
    (`rng.permutation` on a differently-sized array), giving a different
    `p_value` even though nanmean already ignores NaN in the STATISTIC
    itself -- this specifically tests the SHUFFLE structure, not the
    statistic."""
    rng = np.random.default_rng(3)
    n_clusters, per_cluster = 10, 10
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    treated = rng.random(n_clusters * per_cluster) < 0.5
    y = rng.normal(size=n_clusters * per_cluster) + treated * 0.5

    baseline = within_cluster_permutation_test(y, treated, cluster, n_perm=200, seed=7)

    extra_cluster = rng.integers(0, n_clusters, size=15)
    extra_treated = rng.random(15) < 0.5
    y_with_nan = np.concatenate([y, np.full(15, np.nan)])
    treated_with_nan = np.concatenate([treated, extra_treated])
    cluster_with_nan = np.concatenate([cluster, extra_cluster])

    with_nan = within_cluster_permutation_test(
        y_with_nan, treated_with_nan, cluster_with_nan, n_perm=200, seed=7
    )

    assert with_nan.estimate == pytest.approx(baseline.estimate)
    assert with_nan.p_value == pytest.approx(baseline.p_value)
    assert with_nan.n_treated == baseline.n_treated
    assert with_nan.n_control == baseline.n_control
    assert with_nan.n_clusters == baseline.n_clusters


def test_within_cluster_permutation_test_reports_complete_case_counts() -> None:
    y = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
    treated = np.array([True, False, True, False, True, False])
    cluster = np.array([0, 0, 0, 1, 1, 1])
    result = within_cluster_permutation_test(y, treated, cluster, n_perm=50, seed=0)
    # Row index 2 (cluster 0, treated) is dropped for having a NaN outcome,
    # leaving treated={0, 4} and control={1, 3, 5}.
    assert result.n_treated == 2
    assert result.n_control == 3
    assert result.n_clusters == 2


def test_within_cluster_permutation_test_rejects_a_cluster_wiped_out_by_nan() -> None:
    """`_validate_panel_triplet`'s own >=2-cluster floor runs BEFORE the
    NaN-y complete-case filter, so a cluster whose every row has a NaN
    outcome vanishes only AFTER that check already passed. Here cluster 1
    is entirely NaN, leaving only cluster 0 to permute within -- must
    raise, not silently return a p_value from a degenerate single-cluster
    shuffle."""
    y = np.array([1.0, 2.0, np.nan, np.nan])
    treated = np.array([True, False, True, False])
    cluster = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="at least 2 distinct clusters"):
        within_cluster_permutation_test(y, treated, cluster, n_perm=50, seed=0)


def test_cluster_bootstrap_difference_rejects_non_boolean_treated() -> None:
    y = np.arange(10, dtype=float)
    cluster = np.repeat(np.arange(5), 2)
    treated = np.array(["a", "b"] * 5)
    with pytest.raises(TypeError, match="boolean-like"):
        cluster_bootstrap_difference(y, treated, cluster)


def test_cluster_bootstrap_difference_rejects_a_multi_level_treated_column() -> None:
    """Regression test: a 3-arm group-id column ([0, 1, 2]) has a plain
    int dtype and must not silently pass the dtype-kind check and get
    truthiness-cast into a 2-arm split (0 -> False, 1 AND 2 -> True) --
    that would produce a plausible-looking but meaningless treated/
    control split with no error at all."""
    y = np.arange(9, dtype=float)
    cluster = np.repeat(np.arange(3), 3)
    treated = np.array([0, 1, 2] * 3)
    with pytest.raises(ValueError, match=r"only 0/1"):
        cluster_bootstrap_difference(y, treated, cluster)


def test_cluster_bootstrap_difference_rejects_missing_cluster() -> None:
    y = np.arange(10, dtype=float)
    treated = np.array([True, False] * 5)
    cluster = np.array([0, 1, 2, np.nan, 4, 0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="missing"):
        cluster_bootstrap_difference(y, treated, cluster)


def test_cluster_bootstrap_difference_rejects_a_2d_cluster_array() -> None:
    """Regression test: a 2-D ``cluster`` array (e.g. an accidental
    ``(n, 1)`` column) must be rejected up front, not reach
    ``np.unique``/boolean indexing with silently wrong semantics."""
    y = np.arange(10, dtype=float)
    treated = np.array([True, False] * 5)
    cluster = np.repeat(np.arange(5), 2).reshape(-1, 1)
    with pytest.raises(ValueError, match="one-dimensional"):
        cluster_bootstrap_difference(y, treated, cluster)


def test_cluster_bootstrap_difference_rejects_heterogeneous_cluster_label_types() -> (
    None
):
    """Regression test: ``np.unique`` sorts internally and raises a raw,
    uninformative ``TypeError`` for an object-dtype ``cluster`` array
    mixing incomparable label types (e.g. some rows labeled by an int
    fold index, others by a string entity name) -- must be rejected with
    a clear, named error instead."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    treated = np.array([True, False, True, False, True, False])
    cluster = np.array([1, 1, "a", "a", 2, 2], dtype=object)
    with pytest.raises(TypeError, match="mutually comparable type"):
        cluster_bootstrap_difference(y, treated, cluster, n_boot=50, seed=0)


def test_cluster_bootstrap_difference_rejects_infinite_y() -> None:
    """An infinite y value must be rejected rather than silently
    propagating into an infinite estimate and a NaN confidence interval."""
    y = np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    treated = np.array([True, False] * 5)
    cluster = np.repeat(np.arange(5), 2)
    with pytest.raises(ValueError, match="Infinity"):
        cluster_bootstrap_difference(y, treated, cluster)


def test_cluster_bootstrap_difference_rejects_a_single_cluster() -> None:
    """A single cluster has no resampling variation to estimate from --
    must be rejected rather than silently producing a degenerate CI."""
    y = np.arange(10, dtype=float)
    treated = np.array([True, False] * 5)
    cluster = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="at least 2 distinct clusters"):
        cluster_bootstrap_difference(y, treated, cluster)


def test_cluster_bootstrap_difference_all_treated_gives_nan() -> None:
    y = np.arange(10, dtype=float)
    treated = np.ones(10, dtype=bool)
    cluster = np.repeat(np.arange(5), 2)
    result = cluster_bootstrap_difference(y, treated, cluster, n_boot=50, seed=0)
    assert np.isnan(result.estimate)
    assert np.isnan(result.ci_low)
    assert np.isnan(result.ci_high)


def test_cluster_bootstrap_difference_rejects_a_2d_treated_array() -> None:
    """Regression test: a 2-D ``treated`` array (e.g. an accidental
    ``(n, 1)`` column) must be rejected up front, not reach boolean-mask
    indexing with silently wrong semantics and surface as a confusing
    IndexError far from this call."""
    y = np.arange(10, dtype=float)
    treated = np.tile([True, False], 5).reshape(-1, 1)
    cluster = np.repeat(np.arange(5), 2)
    with pytest.raises(ValueError, match="one-dimensional"):
        cluster_bootstrap_difference(y, treated, cluster)


def test_within_cluster_permutation_test_all_treated_gives_nan() -> None:
    y = np.arange(10, dtype=float)
    treated = np.ones(10, dtype=bool)
    cluster = np.repeat(np.arange(5), 2)
    result = within_cluster_permutation_test(y, treated, cluster, n_perm=50, seed=0)
    assert np.isnan(result.estimate)
    assert np.isnan(result.p_value)
    assert result.n_perm == 0


def test_cluster_bootstrap_difference_preserves_within_cluster_dependence() -> None:
    """A naive row-level bootstrap that ignores cluster membership treats
    every row as an independent draw, badly understating the true
    sampling uncertainty whenever variance is dominated by between-cluster
    differences rather than within-cluster noise. cluster_bootstrap_difference
    must resample whole clusters (with replacement), not individual rows --
    proven here by constructing a panel where almost all the variance is
    between clusters (10 clusters, cluster-level treatment assignment) and
    checking the function's CI is dramatically wider than what a naive
    per-row bootstrap on the identical data would give, since a per-row
    resample effectively (and wrongly) treats n=500 as independent when the
    true effective sample size is n=10 clusters."""
    rng = np.random.default_rng(7)
    n_clusters, per_cluster = 10, 50
    cluster_effect = rng.normal(scale=2.0, size=n_clusters)
    noise_scale = 0.01  # within-cluster variance is negligible next to it.
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    # Treatment is assigned per cluster, not per row: every row in a
    # cluster shares its cluster's status, so all information about any
    # treatment effect lives at the cluster level.
    treated_by_cluster = np.arange(n_clusters) < n_clusters // 2
    treated = treated_by_cluster[cluster]
    y = cluster_effect[cluster] + rng.normal(
        scale=noise_scale, size=n_clusters * per_cluster
    )

    boot = cluster_bootstrap_difference(y, treated, cluster, n_boot=1000, seed=0)
    cluster_ci_width = boot.ci_high - boot.ci_low

    # A deliberately naive per-row bootstrap on the identical data, for
    # contrast -- the exact wrong invariant this test guards against.
    naive_rng = np.random.default_rng(0)
    n = len(y)
    naive_draws = np.empty(1000)
    for i in range(1000):
        idx = naive_rng.integers(0, n, size=n)
        naive_draws[i] = y[idx][treated[idx]].mean() - y[idx][~treated[idx]].mean()
    naive_ci_width = float(
        np.percentile(naive_draws, 97.5) - np.percentile(naive_draws, 2.5)
    )

    assert cluster_ci_width > 3 * naive_ci_width


def test_cluster_bootstrap_median_statistic() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(size=400)
    treated = rng.random(400) < 0.4
    cluster = np.repeat(np.arange(20), 20)
    result = cluster_bootstrap_difference(
        y, treated, cluster, statistic="median", n_boot=200, seed=0
    )
    assert result.ci_low < 0 < result.ci_high


def test_invalid_statistic_name_raises() -> None:
    y = np.arange(10, dtype=float)
    treated = np.array([True, False] * 5)
    cluster = np.repeat(np.arange(5), 2)
    with pytest.raises(ValueError, match="statistic"):
        cluster_bootstrap_difference(y, treated, cluster, statistic="mode")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# fama_macbeth_difference
# --------------------------------------------------------------------------- #


def _null_panel(seed: int, n_clusters: int = 30, per_cluster: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_clusters * per_cluster
    return pd.DataFrame(
        {
            "outcome": rng.normal(size=n),
            "treated": rng.random(n) < 0.4,
            "cluster": np.repeat(np.arange(n_clusters), per_cluster),
        }
    )


def test_fama_macbeth_difference_recovers_null() -> None:
    frame = _null_panel(seed=4)
    result = fama_macbeth_difference(frame, "outcome", "treated", "cluster")
    assert result.p_value > 0.05
    assert result.n_clusters == 30


def test_fama_macbeth_difference_detects_a_real_effect() -> None:
    frame = _null_panel(seed=4)
    frame["outcome"] = frame["outcome"] + frame["treated"] * 2.0
    result = fama_macbeth_difference(frame, "outcome", "treated", "cluster")
    assert result.p_value < 0.01
    assert result.mean_diff == pytest.approx(2.0, abs=0.5)
    assert result.share_positive > 0.8


def test_fama_macbeth_difference_tolerates_a_duplicate_caller_index() -> None:
    """A frame built by concatenating several per-cluster frames without
    `ignore_index` (a realistic, undocumented-as-forbidden shape) has
    duplicate index labels across clusters -- a label-based lookup of
    each cluster's own treated mask must not return rows sharing that
    label from a DIFFERENT cluster."""
    fold_a = pd.DataFrame(
        {"outcome": [1.0, 2.0, 3.0], "treated": [True, False, True], "fold": ["A"] * 3}
    )
    fold_b = pd.DataFrame(
        {
            "outcome": [10.0, 20.0, 30.0],
            "treated": [True, False, True],
            "fold": ["B"] * 3,
        }
    )
    frame = pd.concat([fold_a, fold_b])  # duplicate index [0, 1, 2, 0, 1, 2]
    result = fama_macbeth_difference(frame, "outcome", "treated", "fold")
    assert result.n_clusters == 2
    # fold A: mean([1, 3]) - mean([2]) = 0; fold B: mean([10, 30]) - mean([20]) = 0
    assert result.mean_diff == pytest.approx(0.0)


def test_fama_macbeth_difference_missing_column_raises() -> None:
    frame = _null_panel(seed=4)
    with pytest.raises(ValueError, match="missing"):
        fama_macbeth_difference(frame, "not_a_column", "treated", "cluster")


def test_fama_macbeth_difference_rejects_infinite_outcome() -> None:
    """An infinite outcome must be rejected explicitly, not silently
    propagate into an infinite per-cluster difference that
    ``_cluster_t_test``'s own finite-value filter would then drop from
    ``n_clusters`` with no warning at all."""
    frame = _null_panel(seed=4)
    frame.loc[0, "outcome"] = np.inf
    with pytest.raises(ValueError, match="Infinity"):
        fama_macbeth_difference(frame, "outcome", "treated", "cluster")


def test_fama_macbeth_difference_rejects_a_missing_cluster_label() -> None:
    """`frame.groupby(cluster)` silently DROPS a row whose cluster label
    is missing (pandas' own default) -- indistinguishable from that row
    never existing rather than the data-quality issue it is. Must be
    rejected explicitly instead."""
    frame = _null_panel(seed=4)
    frame.loc[0, "cluster"] = np.nan
    with pytest.raises(ValueError, match=r"'cluster'.*missing values"):
        fama_macbeth_difference(frame, "outcome", "treated", "cluster")


def test_fama_macbeth_difference_ignores_rounding_noise_as_significance() -> None:
    """Regression test: two clusters whose per-cluster difference is
    identical up to floating-point rounding (``1.0`` vs
    ``1.0 - 2.22e-16``) must NOT produce an astronomically large t-stat /
    vanishing p-value -- that would be "significance" manufactured from
    machine-epsilon noise, not from the data. ``t_stat``/``p_value`` must
    come back NaN (no real dispersion trustworthy enough to test against),
    while ``mean_diff`` itself still reports the (~1.0) value honestly and
    ``se`` reports its own (honest, if tiny) computed value rather than
    being hidden."""
    epsilon_gap = 2.22e-16
    frame = pd.DataFrame(
        {
            "outcome": [1.0, 0.0, 1.0 - epsilon_gap, 0.0],
            "treated": [True, False, True, False],
            "cluster": [0, 0, 1, 1],
        }
    )
    result = fama_macbeth_difference(frame, "outcome", "treated", "cluster")
    assert result.mean_diff == pytest.approx(1.0)
    assert result.se < 1e-10  # negligible relative to mean_diff, not exactly 0
    assert np.isnan(result.t_stat)
    assert np.isnan(result.p_value)


def test_fama_macbeth_difference_rejects_fewer_than_two_contributing_clusters() -> None:
    """Only one cluster has both a treated and a control row -- the
    across-cluster standard error this test is built on is meaningless
    from a single contributing cluster, so it must raise rather than
    silently return a one-observation "result"."""
    frame = pd.DataFrame(
        {
            "outcome": [1.0, 2.0, 3.0, 4.0],
            "treated": [True, False, True, True],
            "cluster": [0, 0, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="At least 2 clusters"):
        fama_macbeth_difference(frame, "outcome", "treated", "cluster")


def test_fama_macbeth_difference_excludes_an_all_nan_cluster_from_the_count() -> None:
    """Cluster 0 has both arms but an all-NaN outcome in each, so its
    treated-minus-control mean is NaN, not a real difference -- it must not
    be counted toward the 2-contributing-cluster minimum."""
    frame = pd.DataFrame(
        {
            "outcome": [np.nan, np.nan, 1.0, 2.0],
            "treated": [True, False, True, False],
            "cluster": [0, 0, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="At least 2 clusters"):
        fama_macbeth_difference(frame, "outcome", "treated", "cluster")


# --------------------------------------------------------------------------- #
# per_cluster_spearman
# --------------------------------------------------------------------------- #


def test_per_cluster_spearman_recovers_null() -> None:
    rng = np.random.default_rng(5)
    n_clusters, per_cluster = 25, 20
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n_clusters * per_cluster),
            "y": rng.normal(size=n_clusters * per_cluster),
            "cluster": np.repeat(np.arange(n_clusters), per_cluster),
        }
    )
    result = per_cluster_spearman(frame, "x", "y", "cluster")
    assert result.p_value > 0.05


def test_per_cluster_spearman_detects_a_real_correlation() -> None:
    rng = np.random.default_rng(5)
    n_clusters, per_cluster = 25, 20
    x = rng.normal(size=n_clusters * per_cluster)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": x * 2.0 + rng.normal(scale=0.5, size=n_clusters * per_cluster),
            "cluster": np.repeat(np.arange(n_clusters), per_cluster),
        }
    )
    result = per_cluster_spearman(frame, "x", "y", "cluster")
    assert result.p_value < 0.01
    assert result.mean_rho > 0.5


def test_per_cluster_spearman_excludes_clusters_below_min_observations() -> None:
    """A cluster with fewer than ``min_observations`` complete rows must not
    contribute its own (unstable) correlation -- proven here by adding one
    tiny, noise-only cluster to an otherwise strongly-correlated panel and
    checking it is excluded from ``n_clusters`` rather than dragging the
    across-cluster estimate toward its own noise."""
    rng = np.random.default_rng(5)
    n_clusters, per_cluster = 10, 20
    x = rng.normal(size=n_clusters * per_cluster)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": x * 2.0 + rng.normal(scale=0.1, size=n_clusters * per_cluster),
            "cluster": np.repeat(np.arange(n_clusters), per_cluster),
        }
    )
    tiny_cluster = pd.DataFrame(
        {"x": [1.0, -1.0, 2.0], "y": [-1.0, 1.0, -2.0], "cluster": n_clusters}
    )
    combined = pd.concat([frame, tiny_cluster], ignore_index=True)

    result = per_cluster_spearman(combined, "x", "y", "cluster", min_observations=10)
    assert result.n_clusters == n_clusters  # the 3-row cluster excluded
    assert result.mean_rho > 0.9  # unaffected by the tiny cluster's own noise


def test_per_cluster_spearman_rejects_infinite_values() -> None:
    """Regression test: an infinite x/y value is a data-quality issue (e.g.
    an upstream division by zero), not a value a rank correlation should
    silently absorb -- must be rejected explicitly."""
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.inf, 4.0, 5.0] * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
            "cluster": [0] * 5 + [1] * 5,
        }
    )
    with pytest.raises(ValueError, match="Infinity"):
        per_cluster_spearman(frame, "x", "y", "cluster", min_observations=5)


def test_per_cluster_spearman_rejects_a_missing_cluster_label() -> None:
    """`frame.groupby(cluster)` silently DROPS a row whose cluster label
    is missing -- must be rejected explicitly instead."""
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0] * 2,
            "cluster": [0] * 5 + [1] * 5,
        }
    )
    frame.loc[0, "cluster"] = np.nan
    with pytest.raises(ValueError, match=r"'cluster'.*missing values"):
        per_cluster_spearman(frame, "x", "y", "cluster", min_observations=4)


def test_per_cluster_spearman_rejects_fewer_than_two_contributing_clusters() -> None:
    """Only one cluster meets ``min_observations`` -- meaningless to
    average a single cluster's own correlation across clusters."""
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0],
            "y": [5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0],
            "cluster": [0, 0, 0, 0, 0, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="At least 2 clusters"):
        per_cluster_spearman(frame, "x", "y", "cluster", min_observations=5)


# --------------------------------------------------------------------------- #
# clustered_regression
# --------------------------------------------------------------------------- #


def test_clustered_regression_keeps_parameter_names() -> None:
    frame = _null_panel(seed=6)
    frame["regressor"] = np.random.default_rng(6).normal(size=len(frame))
    fit = clustered_regression(
        frame, "outcome", "regressor", fixed_effects="cluster", clusters=("cluster",)
    )
    assert "regressor" in fit.params.index
    assert "const" in fit.params.index
    assert any(name.startswith("cluster_") for name in fit.params.index)


def test_clustered_regression_recovers_null() -> None:
    rng = np.random.default_rng(7)
    n = 600
    frame = pd.DataFrame(
        {
            "outcome": rng.normal(size=n),
            "regressor": rng.normal(size=n),
            "cluster": np.repeat(np.arange(30), 20),
        }
    )
    fit = clustered_regression(frame, "outcome", "regressor", clusters=("cluster",))
    assert fit.pvalues["regressor"] > 0.05


def test_clustered_regression_detects_a_real_effect() -> None:
    rng = np.random.default_rng(7)
    n = 600
    regressor = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "outcome": 3.0 * regressor + rng.normal(scale=0.5, size=n),
            "regressor": regressor,
            "cluster": np.repeat(np.arange(30), 20),
        }
    )
    fit = clustered_regression(frame, "outcome", "regressor", clusters=("cluster",))
    assert fit.pvalues["regressor"] < 0.01
    assert fit.params["regressor"] == pytest.approx(3.0, abs=0.5)


def test_clustered_regression_two_way_clustering() -> None:
    rng = np.random.default_rng(8)
    n = 400
    frame = pd.DataFrame(
        {
            "outcome": rng.normal(size=n),
            "regressor": rng.normal(size=n),
            "fold": np.repeat(np.arange(20), 20),
            "entity": np.tile(np.arange(20), 20),
        }
    )
    fit = clustered_regression(
        frame, "outcome", "regressor", clusters=("fold", "entity")
    )
    assert "regressor" in fit.params.index


def test_clustered_regression_rejects_more_than_two_way_clustering() -> None:
    frame = _null_panel(seed=6)
    frame["regressor"] = 1.0
    frame["extra"] = 1.0
    with pytest.raises(ValueError, match="at most two-way"):
        clustered_regression(
            frame,
            "outcome",
            "regressor",
            clusters=("cluster", "regressor", "extra"),
        )


def test_clustered_regression_requires_at_least_one_cluster_column() -> None:
    frame = _null_panel(seed=6)
    frame["regressor"] = 1.0
    with pytest.raises(ValueError, match="at least one column"):
        clustered_regression(frame, "outcome", "regressor", clusters=())


def test_clustered_regression_rejects_a_bare_string_as_clusters() -> None:
    """Regression test: `clusters` is documented as a sequence of column
    names, e.g. `("fold",)` -- but a bare string is ALSO a valid
    `Sequence[str]` (iterating it yields one character per element), so
    `clusters="fold"` would otherwise be silently accepted as if it named
    four one-character columns instead of raising a clear error."""
    frame = _null_panel(seed=6)
    frame["regressor"] = 1.0
    with pytest.raises(TypeError, match="bare string"):
        clustered_regression(frame, "outcome", "regressor", clusters="cluster")


def test_clustered_regression_rejects_zero_complete_rows() -> None:
    """A regression on zero rows (every row has a missing value in a
    required column) must raise a clear QuantLab error, not a raw
    NumPy/statsmodels one."""
    frame = pd.DataFrame(
        {
            "outcome": [np.nan, np.nan, np.nan],
            "regressor": [1.0, 2.0, np.nan],
            "cluster": [0, 1, 2],
        }
    )
    with pytest.raises(ValueError, match="No complete"):
        clustered_regression(frame, "outcome", "regressor", clusters=("cluster",))


def test_clustered_regression_rejects_infinite_outcome_or_regressor() -> None:
    frame = _null_panel(seed=6)
    frame["regressor"] = np.random.default_rng(6).normal(size=len(frame))
    frame.loc[0, "outcome"] = np.inf
    with pytest.raises(ValueError, match="Infinity"):
        clustered_regression(frame, "outcome", "regressor", clusters=("cluster",))


def test_clustered_regression_rejects_a_duplicate_cluster_column() -> None:
    """The same column named twice in `clusters` would otherwise silently
    reach statsmodels as if it were two independent clustering dimensions,
    when it's really the same one twice."""
    frame = _null_panel(seed=6)
    frame["regressor"] = np.random.default_rng(6).normal(size=len(frame))
    with pytest.raises(ValueError, match="not repeat"):
        clustered_regression(
            frame, "outcome", "regressor", clusters=("cluster", "cluster")
        )


def test_clustered_regression_rejects_a_single_valued_cluster_column() -> None:
    frame = _null_panel(seed=6)
    frame["regressor"] = np.random.default_rng(6).normal(size=len(frame))
    frame["single_cluster"] = 0
    with pytest.raises(ValueError, match="at least 2 are required"):
        clustered_regression(
            frame, "outcome", "regressor", clusters=("single_cluster",)
        )


def test_clustered_regression_rejects_a_rank_deficient_design() -> None:
    """A constant regressor alongside the intercept makes the design
    matrix singular -- OLS is not identified, and statsmodels would
    otherwise fall back to a pseudo-inverse and return a plausible-
    looking but meaningless fit with no error at all."""
    frame = _null_panel(seed=6)
    frame["regressor"] = 1.0
    with pytest.raises(ValueError, match="rank-deficient"):
        clustered_regression(frame, "outcome", "regressor", clusters=("cluster",))


# --------------------------------------------------------------------------- #
# paired_cluster_bootstrap_sharpe_difference
# --------------------------------------------------------------------------- #


def test_paired_cluster_bootstrap_sharpe_difference_recovers_null() -> None:
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2020-01-01", periods=600)
    a = pd.Series(rng.normal(0.0005, 0.01, size=600), index=idx)
    b = pd.Series(rng.normal(0.0005, 0.01, size=600), index=idx)
    cluster = pd.Series(np.repeat(np.arange(30), 20), index=idx)
    result = paired_cluster_bootstrap_sharpe_difference(
        a, b, cluster, n_boot=300, seed=0, periods_per_year=252
    )
    assert result.ci_low < 0 < result.ci_high
    assert result.p_value_two_sided > 0.05


def test_paired_cluster_bootstrap_sharpe_difference_detects_a_real_effect() -> None:
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2020-01-01", periods=600)
    a = pd.Series(rng.normal(0.002, 0.01, size=600), index=idx)
    b = pd.Series(rng.normal(0.0, 0.01, size=600), index=idx)
    cluster = pd.Series(np.repeat(np.arange(30), 20), index=idx)
    result = paired_cluster_bootstrap_sharpe_difference(
        a, b, cluster, n_boot=300, seed=0, periods_per_year=252
    )
    assert result.ci_low > 0
    assert result.p_value_two_sided < 0.05


def test_paired_cluster_bootstrap_sharpe_difference_subtracts_risk_free_rate() -> None:
    """A nonzero ``risk_free_rate`` must materially change the estimate
    versus the default (``0.0``)."""
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2020-01-01", periods=600)
    a = pd.Series(rng.normal(0.0008, 0.01, size=600), index=idx)
    b = pd.Series(rng.normal(0.0003, 0.01, size=600), index=idx)
    cluster = pd.Series(np.repeat(np.arange(30), 20), index=idx)
    zero_rf = paired_cluster_bootstrap_sharpe_difference(
        a, b, cluster, n_boot=100, seed=0, periods_per_year=252, risk_free_rate=0.0
    )
    with_rf = paired_cluster_bootstrap_sharpe_difference(
        a, b, cluster, n_boot=100, seed=0, periods_per_year=252, risk_free_rate=0.05
    )
    assert zero_rf.estimate != pytest.approx(with_rf.estimate)


def test_paired_cluster_bootstrap_sharpe_difference_rejects_a_bool_risk_free_rate() -> (
    None
):
    """`True`/`False` are technically `Real` in Python but must not be
    silently accepted as `1.0`/`0.0` -- an annual risk-free rate of
    "100%" from a stray `True` would go undetected."""
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2020-01-01", periods=40)
    a = pd.Series(rng.normal(0.001, 0.01, size=40), index=idx)
    b = pd.Series(rng.normal(0.0, 0.01, size=40), index=idx)
    cluster = pd.Series(np.repeat(np.arange(4), 10), index=idx)
    with pytest.raises(TypeError, match="real number"):
        paired_cluster_bootstrap_sharpe_difference(
            a, b, cluster, n_boot=10, seed=0, risk_free_rate=True
        )


def test_paired_cluster_bootstrap_sharpe_difference_default_periods_per_year() -> None:
    """Omitting periods_per_year must annualise with TRADING_DAYS_PER_YEAR
    (252), identically to passing it explicitly. Uses non-degenerate
    returns rather than an all-zero series: a zero-variance series yields
    a NaN Sharpe (see the sharpe() helper's own docstring), not a usable
    "predictable estimate" for this comparison."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=40)
    a = pd.Series(rng.normal(0.001, 0.01, size=40), index=idx)
    b = pd.Series(rng.normal(0.0, 0.01, size=40), index=idx)
    cluster = pd.Series(np.repeat(np.arange(4), 10), index=idx)
    default = paired_cluster_bootstrap_sharpe_difference(
        a, b, cluster, n_boot=10, seed=0
    )
    explicit = paired_cluster_bootstrap_sharpe_difference(
        a, b, cluster, n_boot=10, seed=0, periods_per_year=252
    )
    assert default.estimate == pytest.approx(explicit.estimate)


def test_paired_cluster_bootstrap_sharpe_difference_zero_variance_is_nan_not_zero() -> (
    None
):
    """A constant (zero-variance) series has an undefined Sharpe ratio --
    the observed estimate and every bootstrap draw touching it must come
    back NaN, never a fabricated 0.0 that would misrepresent "no evidence"
    as "confirmed no difference"."""
    idx = pd.bdate_range("2020-01-01", periods=40)
    a = pd.Series(0.0, index=idx)
    b = pd.Series(0.0, index=idx)
    cluster = pd.Series(np.repeat(np.arange(4), 10), index=idx)
    result = paired_cluster_bootstrap_sharpe_difference(
        a, b, cluster, n_boot=10, seed=0
    )
    assert np.isnan(result.estimate)
    assert np.isnan(result.ci_low)
    assert np.isnan(result.ci_high)
    assert np.isnan(result.p_value_two_sided)


def test_paired_cluster_bootstrap_sharpe_difference_rejects_a_duplicate_index() -> None:
    """A duplicate-labelled index must be rejected outright -- a
    label-based `pd.concat(..., axis=1)` alignment against one would
    silently misalign or cross-product rows sharing a label."""
    idx = pd.bdate_range("2020-01-01", periods=10).append(
        pd.bdate_range("2020-01-01", periods=1)
    )
    a = pd.Series(np.arange(11, dtype=float), index=idx)
    b = pd.Series(np.arange(11, dtype=float), index=idx)
    cluster = pd.Series(np.repeat(np.arange(3), 4)[:11], index=idx)
    with pytest.raises(ValueError, match="duplicate labels"):
        paired_cluster_bootstrap_sharpe_difference(a, b, cluster, n_boot=10, seed=0)


def test_paired_cluster_bootstrap_sharpe_difference_rejects_infinite_returns() -> None:
    """An infinite return must be rejected explicitly, not silently
    propagate into an infinite/NaN Sharpe ratio that the `np.isfinite`
    filter on `draws` would then drop from the bootstrap distribution
    with no warning."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    a = pd.Series([np.inf] + [0.001] * 9, index=idx)
    b = pd.Series(0.0005, index=idx)
    cluster = pd.Series(np.repeat(np.arange(2), 5), index=idx)
    with pytest.raises(ValueError, match="Infinity"):
        paired_cluster_bootstrap_sharpe_difference(a, b, cluster, n_boot=10, seed=0)


def test_paired_cluster_bootstrap_sharpe_difference_rejects_a_single_cluster() -> None:
    """A single cluster has no resampling variation to estimate from --
    every draw would resample that same one cluster, producing a
    degenerate zero-width CI that looks confident but reflects no
    genuine uncertainty."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    a = pd.Series(np.linspace(0.001, 0.01, 10), index=idx)
    b = pd.Series(np.linspace(0.0005, 0.005, 10), index=idx)
    cluster = pd.Series(0, index=idx)
    with pytest.raises(ValueError, match="at least 2 distinct clusters"):
        paired_cluster_bootstrap_sharpe_difference(a, b, cluster, n_boot=10, seed=0)


# --------------------------------------------------------------------------- #
# cliffs_delta
# --------------------------------------------------------------------------- #


def test_cliffs_delta_identical_distributions_near_zero() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    assert abs(cliffs_delta(x, y)) < 0.1


def test_cliffs_delta_fully_separated_distributions() -> None:
    x = np.arange(10, 20, dtype=float)
    y = np.arange(0, 10, dtype=float)
    assert cliffs_delta(x, y) == pytest.approx(1.0)
    assert cliffs_delta(y, x) == pytest.approx(-1.0)


def test_cliffs_delta_rejects_2d_input() -> None:
    """A 2-D array (e.g. an accidental ``(n, 1)`` column, or a genuine
    multi-column matrix) must be rejected, not silently flattened through
    Mann-Whitney U against the documented one-dimensional contract."""
    x = np.arange(10, dtype=float).reshape(-1, 2)
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="one-dimensional"):
        cliffs_delta(x, y)


def test_cliffs_delta_empty_sample_is_nan() -> None:
    assert np.isnan(cliffs_delta(np.array([]), np.array([1.0, 2.0])))


def test_cliffs_delta_rejects_infinity_instead_of_silently_dropping_it() -> None:
    """Regression test: `np.isfinite`-based filtering would silently drop
    an Infinity alongside NaN, as if it never existed --
    `cliffs_delta([1.0, np.inf], [0.0, 2.0])` must not quietly return the
    delta for `[1.0]` vs `[0.0, 2.0]` as if the infinite value were never
    there. Infinity is a data-quality issue here, not missing data."""
    with pytest.raises(ValueError, match="Infinity"):
        cliffs_delta([1.0, np.inf], [0.0, 2.0])
    with pytest.raises(ValueError, match="Infinity"):
        cliffs_delta([1.0, 2.0], [0.0, -np.inf])
    # NaN alone is still silently dropped, unaffected by this fix.
    assert cliffs_delta([1.0, np.nan], [1.0]) == pytest.approx(0.0)
