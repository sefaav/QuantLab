"""Dependence-aware inference for a panel of (entity, cluster) observations.

`bootstrap_returns`/`monte_carlo_permutation` (see :mod:`quantlab.validation.
bootstrap`/:mod:`quantlab.validation.robustness`) make inferential statements
about *one* stitched return series. A multi-asset walk-forward study instead
produces a *panel*: one row per entity (e.g. a pair, an asset) per fold, where
rows sharing a fold share a market environment (cross-sectional dependence)
and rows sharing an entity recur across folds (the same entity is not an
independent draw each fold). Treating every row as independent -- a plain
t-test or OLS with i.i.d. standard errors -- understates uncertainty and can
manufacture significance out of pure cross-sectional co-movement within a
fold. Every function here instead resamples, permutes, or clusters at ONE
caller-chosen dependence level: most (the bootstrap/permutation/
Fama-MacBeth/Spearman/paired-Sharpe functions) accept a single ``cluster``
label and correct for dependence along that one dimension only; only
:func:`clustered_regression` supports two-way clustering (e.g. fold AND
entity simultaneously). None of them automatically correct for dependence
BETWEEN clusters themselves (e.g. serial correlation between temporally
adjacent or overlapping walk-forward folds) -- each function's own
docstring states exactly what it does and does not correct for.

None of these establish causality. They quantify whether an observed
association between a panel column (e.g. a formation-period eligibility
flag) and an outcome (e.g. out-of-sample return) is larger than dependence-
respecting resampling variation would typically produce, under the single
dependence structure the caller supplied -- see each function's own
docstring for its exact null hypothesis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import stats

from quantlab.constants import TRADING_DAYS_PER_YEAR
from quantlab.risk._validation import finite_real, nonnegative_int, positive_int

_STATISTIC_FUNCS: dict[str, Any] = {"mean": np.nanmean, "median": np.nanmedian}

Statistic = Literal["mean", "median"]


@dataclass(frozen=True)
class ClusterDifferenceResult:
    """:func:`cluster_bootstrap_difference`'s own result. See its docstring."""

    estimate: float
    ci_low: float
    ci_high: float
    n_treated: int
    n_control: int
    n_clusters: int
    n_boot_valid: int


@dataclass(frozen=True)
class PermutationTestResult:
    """:func:`within_cluster_permutation_test`'s own result. See its docstring."""

    estimate: float
    p_value: float
    n_perm: int
    # The complete-case counts (after dropping a NaN-``y`` row entirely,
    # never merely excluding it from the statistic while still letting it
    # absorb a shuffled label) the permutation actually ran over -- report
    # these alongside the p-value, since a caller can otherwise not tell
    # how much of the panel a NaN-heavy outcome column silently excluded.
    n_treated: int
    n_control: int
    n_clusters: int


@dataclass(frozen=True)
class FamaMacBethResult:
    """:func:`fama_macbeth_difference`'s own result. See its docstring."""

    mean_diff: float
    se: float
    t_stat: float
    p_value: float
    n_clusters: int
    share_positive: float


@dataclass(frozen=True)
class ClusterSpearmanResult:
    """:func:`per_cluster_spearman`'s own result. See its docstring."""

    mean_rho: float
    se: float
    t_stat: float
    p_value: float
    n_clusters: int


@dataclass(frozen=True)
class PairedSharpeDifferenceResult:
    """:func:`paired_cluster_bootstrap_sharpe_difference`'s own result.

    See its docstring.
    """

    estimate: float
    ci_low: float
    ci_high: float
    p_value_two_sided: float
    n_clusters: int


def _statistic_func(statistic: str) -> Any:
    """Return the NaN-aware reducer for ``statistic``, or raise."""
    try:
        return _STATISTIC_FUNCS[statistic]
    except KeyError:
        raise ValueError(
            f"statistic must be one of {sorted(_STATISTIC_FUNCS)}, got {statistic!r}."
        ) from None


def _validate_boolean_like(treated: object, *, name: str = "treated") -> np.ndarray:
    """Return ``treated`` as a genuine boolean array, or raise.

    Rejects a dtype that cannot mean "boolean" at all (strings, arbitrary
    objects) rather than silently truthiness-casting it; a genuine
    bool/int/float array is still accepted via standard 0/1 <-> False/True
    casting, matching how a caller like `rng.random(n) < p` already
    produces one. Also rejects missing values outright -- a missing label
    is neither treated nor control, and must never be silently coerced
    into one via e.g. `.fillna(False)`.

    A non-bool dtype passing the dtype-kind check alone is not enough: a
    multi-level group-id column (e.g. ``[0, 1, 2]`` for a 3-arm design)
    has an ``int`` dtype and would otherwise reach ``.astype(bool)``
    unchallenged, silently truthiness-casting every non-zero level (1 AND
    2) into ``True`` and producing a plausible-looking but meaningless
    treated/control split with no error. Every non-bool value must
    therefore be exactly ``0`` or ``1``.
    """
    raw = np.asarray(treated)
    if raw.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, got shape {raw.shape} -- a "
            "2-D column (e.g. an (n, 1) array) would otherwise reach "
            "boolean-mask indexing with silently wrong semantics and "
            "surface as a confusing IndexError far from this call."
        )
    if raw.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be a boolean-like array, got dtype {raw.dtype}.")
    if raw.dtype.kind == "f" and np.isnan(raw).any():
        raise ValueError(f"{name} must not contain missing values.")
    if raw.dtype.kind != "b" and not np.isin(raw, (0, 1)).all():
        raise ValueError(
            f"{name} must contain only 0/1 (or boolean) values -- got a "
            "numeric value outside {0, 1}, which cannot be interpreted as "
            "a two-arm treated/control split."
        )
    return raw.astype(bool)


def _unique_cluster_labels(cluster_arr: np.ndarray) -> np.ndarray:
    """``np.unique(cluster_arr)``, with a clear error on heterogeneous labels.

    ``np.unique`` sorts internally, which raises a raw, uninformative
    ``TypeError`` (``'<' not supported between instances of 'str' and
    'int'``) for an ``object``-dtype array mixing incomparable label
    types (e.g. some rows labeled by an int fold index, others by a
    string entity name) -- a caller error this function names plainly
    instead.
    """
    try:
        return np.unique(cluster_arr)
    except TypeError as exc:
        raise TypeError(
            "cluster labels must be of a single, mutually comparable type "
            f"(e.g. all int, or all str): {exc}"
        ) from exc


def _validate_panel_triplet(
    y: object, treated: object, cluster: object, *, min_clusters: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return ``(y, treated, cluster)`` as aligned 1-D arrays.

    ``y`` may contain NaN (every statistic below is computed with a
    NaN-aware reducer) but not Infinity, which would silently propagate
    into an infinite ``estimate`` and a NaN confidence interval with no
    warning. ``treated``/``cluster`` must not contain missing values,
    since a missing treatment label or cluster assignment cannot be
    silently dropped from a resampling scheme without changing cluster
    membership. ``cluster`` must be one-dimensional -- a 2-D array (e.g.
    a caller accidentally passing a ``(n, 1)`` column instead of a flat
    label array) would otherwise reach ``np.unique``/boolean indexing
    with silently wrong semantics rather than a clear error.
    ``min_clusters`` (default 2) rejects a panel with too few distinct
    clusters. A single-cluster resample IS technically definable (e.g. a
    within-cluster permutation), but QuantLab requires at least two
    clusters as a conservative inferential-safety rule, not because fewer
    is mathematically undefined -- one cluster has no BETWEEN-cluster
    resampling variation to estimate from, making any resulting interval
    or p-value a poor basis for inference regardless of how many rows that
    one cluster has. This is only a minimum, not a claim that inference
    from just two or three clusters is reliable either -- cluster-robust
    standard errors and bootstrap CIs are known to be fragile with only a
    handful of clusters regardless of how many rows each one has.
    """
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if np.isinf(y_arr).any():
        raise ValueError("y must not contain Infinity.")
    treated_arr = _validate_boolean_like(treated)
    cluster_arr = np.asarray(cluster)
    if cluster_arr.ndim != 1:
        raise ValueError("cluster must be one-dimensional.")
    if not (len(y_arr) == len(treated_arr) == len(cluster_arr)):
        raise ValueError("y, treated and cluster must have the same length.")
    if len(y_arr) == 0:
        raise ValueError("y must not be empty.")
    if pd.isna(cluster_arr).any():
        raise ValueError("cluster must not contain missing values.")
    n_clusters = len(_unique_cluster_labels(cluster_arr))
    if n_clusters < min_clusters:
        raise ValueError(
            f"cluster must contain at least {min_clusters} distinct clusters "
            f"for cluster-level resampling to be well-defined, got {n_clusters}."
        )
    return y_arr, treated_arr, cluster_arr


def _group_difference(y: np.ndarray, treated: np.ndarray, func: Any) -> float:
    """``func(y[treated]) - func(y[~treated])``, or NaN if either side is empty."""
    if treated.sum() == 0 or (~treated).sum() == 0:
        return float("nan")
    return float(func(y[treated]) - func(y[~treated]))


def cluster_bootstrap_difference(
    y: object,
    treated: object,
    cluster: object,
    *,
    statistic: Statistic = "mean",
    n_boot: int = 2000,
    seed: int = 0,
) -> ClusterDifferenceResult:
    """Cluster-bootstrap CI for ``stat(y | treated) - stat(y | control)``.

    Null hypothesis: the treated and control groups have the same
    population ``statistic`` -- rejected at the 95% level when the
    returned CI excludes zero. Whole clusters (e.g. walk-forward folds)
    are resampled with replacement, preserving cross-sectional dependence
    among rows sharing a cluster; this is the panel analogue of
    :func:`quantlab.validation.bootstrap.bootstrap_returns`'s block
    resampling, resampling entire clusters instead of contiguous time
    blocks.

    Args:
        y: Outcome values, one per row. May contain NaN.
        treated: Boolean treatment indicator, one per row.
        cluster: Cluster label, one per row (e.g. fold index).
        statistic: ``"mean"`` or ``"median"``.
        n_boot: Number of cluster-resampled draws.
        seed: Seed for `numpy.random.default_rng`.

    Returns:
        ``estimate`` (the observed difference), ``ci_low``/``ci_high`` (a
        95% percentile interval over ``n_boot`` draws whose own treated
        and control groups were both non-empty -- a draw where one side
        is empty contributes no sample rather than a fabricated value),
        ``n_treated``, ``n_control``, ``n_clusters``, ``n_boot_valid``
        (how many of the ``n_boot`` draws actually survived filtering --
        report this alongside the CI, since a CI computed from very few
        surviving draws is unreliable even when it looks numerically
        fine). ``estimate`` and the CI bounds are NaN when the full panel
        has no treated or no control rows at all.
    """
    func = _statistic_func(statistic)
    n_boot = positive_int(n_boot, name="n_boot")
    seed = nonnegative_int(seed, name="seed")
    y_arr, treated_arr, cluster_arr = _validate_panel_triplet(y, treated, cluster)

    # Complete-case filter: a NaN-y row is dropped entirely here, never
    # left counted toward a cluster's own membership -- mirrors
    # `within_cluster_permutation_test`'s identical filter. Without this,
    # a cluster whose EVERY row has a NaN outcome (contributing nothing
    # real) still gets counted in `n_clusters` and in the resampling
    # population `rng.choice` draws from below, silently biasing both
    # the reported cluster/treated/control counts and the effective
    # resampling variance -- `func` (already NaN-aware) tolerates a NaN
    # row mixed in with real ones within a draw just fine, so dropping
    # it here changes no valid computation's result, only removes
    # clusters/rows that were never informative to begin with.
    complete = ~np.isnan(y_arr)
    y_arr = y_arr[complete]
    treated_arr = treated_arr[complete]
    cluster_arr = cluster_arr[complete]

    clusters = np.unique(cluster_arr)
    # `_validate_panel_triplet`'s own >=2-cluster floor is checked BEFORE
    # the complete-case filter above -- a cluster can vanish entirely
    # there (every one of its rows had a NaN y), so the count must be
    # re-checked on the complete case: resampling from a single cluster
    # has zero variance, so a bootstrap CI is not well-defined at all.
    if len(clusters) < 2:
        raise ValueError(
            "After dropping NaN-y rows, at least 2 distinct clusters must "
            "remain for cluster-level resampling to be well-defined; got "
            f"{len(clusters)}."
        )
    members = {c: np.flatnonzero(cluster_arr == c) for c in clusters}
    rng = np.random.default_rng(seed)

    def diff(idx: np.ndarray) -> float:
        return _group_difference(y_arr[idx], treated_arr[idx], func)

    observed = diff(np.arange(len(y_arr)))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([members[c] for c in chosen])
        draws[i] = diff(idx)
    n_boot_valid = int(np.isfinite(draws).sum())
    draws = draws[np.isfinite(draws)]
    ci_low = float(np.percentile(draws, 2.5)) if len(draws) else float("nan")
    ci_high = float(np.percentile(draws, 97.5)) if len(draws) else float("nan")
    return ClusterDifferenceResult(
        estimate=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        n_treated=int(treated_arr.sum()),
        n_control=int((~treated_arr).sum()),
        n_clusters=len(clusters),
        n_boot_valid=n_boot_valid,
    )


def within_cluster_permutation_test(
    y: object,
    treated: object,
    cluster: object,
    *,
    statistic: Statistic = "mean",
    n_perm: int = 2000,
    seed: int = 0,
) -> PermutationTestResult:
    """Two-sided permutation p-value, shuffling treatment labels within cluster.

    Null hypothesis: treatment carries no information about ``y``, so
    within each cluster the treated/control labels are exchangeable
    among that cluster's rows. The number of treated rows per cluster is
    held fixed under every permutation (only labels *within* a cluster
    are shuffled), which is what makes this the permutation analogue of
    :func:`cluster_bootstrap_difference`'s resampling scheme.

    Args:
        y: Outcome values, one per row. May contain NaN -- a NaN-``y`` row
            is dropped ENTIRELY (from ``y``/``treated``/``cluster`` alike)
            before any grouping or permutation happens, never merely
            excluded from the statistic while still eligible to receive a
            shuffled label: leaving it in the shuffle pool would let the
            number of treated rows WITH a real outcome fluctuate between
            permutations even though the raw treated/control label count
            per cluster stays fixed, silently distorting the null
            distribution relative to a clean complete-case design.
        treated: Boolean treatment indicator, one per row.
        cluster: Cluster label, one per row.
        statistic: ``"mean"`` or ``"median"``.
        n_perm: Number of label permutations.
        seed: Seed for `numpy.random.default_rng`.

    Returns:
        ``estimate`` (the observed difference), ``p_value`` (two-sided,
        with the standard ``(count + 1) / (valid + 1)`` finite-sample
        correction over permutations whose own treated/control groups
        were both non-empty), ``n_perm`` (how many permutations were
        valid), and ``n_treated``/``n_control``/``n_clusters`` (the
        complete-case counts actually used, after dropping NaN-``y``
        rows). ``estimate`` and ``p_value`` are NaN when the complete-case
        panel has no treated or no control rows at all.
    """
    func = _statistic_func(statistic)
    n_perm = positive_int(n_perm, name="n_perm")
    seed = nonnegative_int(seed, name="seed")
    y_arr, treated_arr, cluster_arr = _validate_panel_triplet(y, treated, cluster)

    # Complete-case filter: a NaN-y row is dropped entirely here, never
    # left in the shuffle pool (see the `y` argument's own doc above).
    complete = ~np.isnan(y_arr)
    y_arr = y_arr[complete]
    treated_arr = treated_arr[complete]
    cluster_arr = cluster_arr[complete]
    n_treated = int(treated_arr.sum())
    n_control = int((~treated_arr).sum())
    n_clusters = len(np.unique(cluster_arr)) if len(cluster_arr) else 0
    # _validate_panel_triplet's own >=2-cluster floor is checked BEFORE
    # this filter -- a cluster can vanish entirely here (every one of its
    # rows had a NaN y), so the count must be re-checked on the complete
    # case: within-cluster label shuffling is not well-defined with fewer
    # than 2 clusters actually left to permute within.
    if n_clusters < 2:
        raise ValueError(
            "After dropping NaN-y rows, at least 2 distinct clusters must "
            "remain for within-cluster label permutation to be "
            f"well-defined; got {n_clusters}."
        )

    groups = [np.flatnonzero(cluster_arr == c) for c in np.unique(cluster_arr)]
    rng = np.random.default_rng(seed)

    observed = _group_difference(y_arr, treated_arr, func)
    if not np.isfinite(observed):
        return PermutationTestResult(
            estimate=observed,
            p_value=float("nan"),
            n_perm=0,
            n_treated=n_treated,
            n_control=n_control,
            n_clusters=n_clusters,
        )

    count = 0
    valid = 0
    for _ in range(n_perm):
        permuted = treated_arr.copy()
        for idx in groups:
            permuted[idx] = rng.permutation(treated_arr[idx])
        value = _group_difference(y_arr, permuted, func)
        if np.isfinite(value):
            valid += 1
            if abs(value) >= abs(observed):
                count += 1
    p_value = (count + 1) / (valid + 1) if valid else float("nan")
    return PermutationTestResult(
        estimate=observed,
        p_value=p_value,
        n_perm=valid,
        n_treated=n_treated,
        n_control=n_control,
        n_clusters=n_clusters,
    )


def _cluster_t_test(diffs: Sequence[float]) -> dict[str, float]:
    """Mean/SE/t/p/share-positive over one value per cluster.

    ``se`` is treated as degenerate (no real dispersion to test) not just
    when it is exactly ``0.0``, but whenever it is at or below a noise
    floor scaled to the data's own magnitude
    (``max(|diffs|) * machine-epsilon * 100``): two clusters whose values
    differ only by floating-point rounding (e.g. ``1.0`` and
    ``1.0 - 2.22e-16``) would otherwise produce an astronomically large,
    entirely spurious ``t_stat``/tiny ``p_value`` -- manufactured
    "significance" from rounding noise, not from the data.
    """
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    mean = float(arr.mean()) if n else float("nan")
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    noise_floor = float(np.abs(arr).max()) * np.finfo(float).eps * 100.0 if n else 0.0
    t_stat = mean / se if n > 1 and se > noise_floor else float("nan")
    p_value = (
        float(2 * stats.t.sf(abs(t_stat), df=n - 1))
        if n > 1 and np.isfinite(t_stat)
        else float("nan")
    )
    return {
        "mean": mean,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_clusters": n,
        "share_positive": float((arr > 0).mean()) if n else float("nan"),
    }


def fama_macbeth_difference(
    frame: pd.DataFrame, outcome: str, treated: str, cluster: str
) -> FamaMacBethResult:
    """Average within-cluster treated-minus-control difference, Fama-MacBeth style.

    Null hypothesis: the across-cluster mean of the per-cluster
    treated-minus-control difference is zero. Each cluster contributes
    exactly one difference regardless of how many rows it has, so the
    across-cluster standard error is computed from ``n_clusters``
    observations rather than treating every ROW as its own independent
    observation, which would understate it whenever rows within a
    cluster co-move -- see this module's own docstring for what this
    does and does not correct for (in particular, dependence BETWEEN
    clusters is not).

    Args:
        frame: Panel with one row per (entity, cluster).
        outcome: Column of outcome values (may contain NaN).
        treated: Boolean (or boolean-like) column.
        cluster: Column identifying the cluster (e.g. fold).

    Returns:
        ``mean_diff``, ``se``, ``t_stat``, ``p_value``, ``n_clusters``
        (clusters contributing a finite difference, i.e. with both
        treated and control rows present), ``share_positive``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    for column in (outcome, treated, cluster):
        if column not in frame.columns:
            raise ValueError(f"frame is missing required column {column!r}.")
    # Same rejection _validate_panel_triplet applies to `y` elsewhere in
    # this module: an infinite outcome would silently propagate into an
    # infinite per-cluster difference, which `_cluster_t_test`'s own
    # `np.isfinite` filter would then just as silently DROP -- an entire
    # cluster vanishing from `n_clusters` with no warning, rather than a
    # clear error at the source.
    if np.isinf(frame[outcome].to_numpy(dtype=float)).any():
        raise ValueError(f"column {outcome!r} must not contain Infinity.")
    # `frame.groupby(cluster)` silently DROPS a row whose cluster label is
    # missing (pandas' own default) -- indistinguishable from that row
    # never having existed, rather than the data-quality issue it is.
    if frame[cluster].isna().any():
        raise ValueError(f"column {cluster!r} must not contain missing values.")
    # Reset to a guaranteed-unique 0..n-1 index first: `frame`'s own index
    # is caller-supplied and may contain duplicate labels (e.g. built by
    # `pd.concat`-ing several per-cluster frames without `ignore_index`).
    # `treated_mask.loc[part.index]` below is a LABEL lookup -- against a
    # duplicate label it returns every row sharing that label across ALL
    # clusters, not just this one, silently producing a longer-than-`part`
    # mask (raising downstream, or on flukily-matching lengths, propagating
    # to another cluster's rows entirely). Positions are never ambiguous.
    frame = frame.reset_index(drop=True)
    # Validate once, up front (dtype + no missing labels), then reuse the
    # same boolean Series inside the loop below via its aligned index.
    treated_mask = pd.Series(
        _validate_boolean_like(frame[treated].to_numpy(), name=repr(treated)),
        index=frame.index,
    )
    diffs: list[float] = []
    for _, part in frame.groupby(cluster):
        mask: pd.Series = treated_mask.loc[part.index].astype(bool)
        if mask.sum() > 0 and (~mask).sum() > 0:
            treated_values = part.loc[mask.to_numpy(), outcome]
            control_values = part.loc[(~mask).to_numpy(), outcome]
            diff = float(treated_values.mean() - control_values.mean())
            if np.isfinite(diff):
                diffs.append(diff)
    if len(diffs) < 2:
        raise ValueError(
            "At least 2 clusters must contribute a difference (both a "
            "treated and a control row) for the across-cluster standard "
            f"error to be well-defined at all; got {len(diffs)}."
        )
    result = _cluster_t_test(diffs)
    return FamaMacBethResult(
        mean_diff=result["mean"],
        se=result["se"],
        t_stat=result["t_stat"],
        p_value=result["p_value"],
        n_clusters=int(result["n_clusters"]),
        share_positive=result["share_positive"],
    )


def per_cluster_spearman(
    frame: pd.DataFrame, x: str, y: str, cluster: str, *, min_observations: int = 10
) -> ClusterSpearmanResult:
    """Fama-MacBeth-style average of per-cluster Spearman correlations.

    Null hypothesis: the across-cluster mean rank correlation between
    ``x`` and ``y`` is zero. A cluster contributes only when it has at
    least ``min_observations`` complete (non-missing) rows, since a
    rank correlation from very few points is unstable.

    Args:
        frame: Panel with one row per (entity, cluster).
        x: First numeric column.
        y: Second numeric column.
        cluster: Column identifying the cluster.
        min_observations: Minimum complete rows for a cluster to
            contribute its own Spearman correlation.

    Returns:
        ``mean_rho``, ``se``, ``t_stat``, ``p_value``, ``n_clusters``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    for column in (x, y, cluster):
        if column not in frame.columns:
            raise ValueError(f"frame is missing required column {column!r}.")
    if np.isinf(frame[[x, y]].to_numpy(dtype=float)).any():
        raise ValueError(
            f"columns {x!r}/{y!r} must not contain Infinity -- likely a "
            "data-quality issue (e.g. a division by zero upstream) rather "
            "than a value a rank correlation should silently absorb."
        )
    # `frame.groupby(cluster)` silently DROPS a row whose cluster label is
    # missing (pandas' own default) -- indistinguishable from that row
    # never having existed, rather than the data-quality issue it is.
    if frame[cluster].isna().any():
        raise ValueError(f"column {cluster!r} must not contain missing values.")
    min_observations = positive_int(min_observations, name="min_observations")
    rhos: list[float] = []
    for _, part in frame.groupby(cluster):
        clean = part[[x, y]].dropna()
        if len(clean) >= min_observations:
            spearman_result = stats.spearmanr(clean[x].to_numpy(), clean[y].to_numpy())
            rho = float(cast(Any, spearman_result).statistic)
            if np.isfinite(rho):
                rhos.append(rho)
    if len(rhos) < 2:
        raise ValueError(
            "At least 2 clusters must contribute a rank correlation (at "
            f"least {min_observations} complete rows each) for the "
            f"across-cluster standard error to be well-defined at all; "
            f"got {len(rhos)}."
        )
    result = _cluster_t_test(rhos)
    return ClusterSpearmanResult(
        mean_rho=result["mean"],
        se=result["se"],
        t_stat=result["t_stat"],
        p_value=result["p_value"],
        n_clusters=int(result["n_clusters"]),
    )


def clustered_regression(
    frame: pd.DataFrame,
    outcome: str,
    regressor: str,
    *,
    fixed_effects: str | None = None,
    clusters: Sequence[str] = (),
) -> Any:
    """OLS of ``outcome`` on ``regressor`` with one- or two-way clustered SEs.

    Args:
        frame: Panel with one row per (entity, cluster).
        outcome: Dependent-variable column.
        regressor: Independent-variable column.
        fixed_effects: Optional column whose levels get dummy fixed
            effects (all but one level; the omitted level is absorbed
            into the intercept), e.g. a cluster column to absorb common
            shocks shared by every entity in the same cluster.
        clusters: One or two column names to cluster standard errors by
            (e.g. ``(cluster,)`` or ``(cluster, entity)`` for two-way
            clustering). Must be non-empty.

    Returns:
        The fitted `statsmodels.regression.linear_model.
        RegressionResultsWrapper` (untyped here: statsmodels ships no
        type stubs). The constant column is inserted directly
        (``design.insert(0, "const", 1.0)``) rather than via
        ``sm.add_constant`` -- not a runtime data-loss concern (it does
        preserve a DataFrame's column names at runtime), but its type
        stubs are typed as returning ``ndarray`` regardless of input
        type, which would make mypy treat the result as unlabelled from
        here on even though it is not.
    """
    import statsmodels.api as sm

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    # A bare string is itself a `Sequence[str]` (iterating it yields one
    # character per element), so `clusters="fold"` would otherwise be
    # silently accepted as if it named four one-character columns ("f",
    # "o", "l", "d") instead of raising -- explicitly rejected here rather
    # than left to whatever confusing KeyError that produces downstream.
    if isinstance(clusters, str):
        raise TypeError(
            "clusters must be a sequence of column names, e.g. (clusters,) "
            f"or (clusters, entity) -- got a bare string {clusters!r}, which "
            "would otherwise be silently iterated character by character."
        )
    required = {outcome, regressor, *clusters}
    if fixed_effects is not None:
        required.add(fixed_effects)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required column(s): {sorted(missing)}.")
    if not clusters:
        raise ValueError("clusters must name at least one column.")
    if len(clusters) > 2:
        raise ValueError("clusters supports at most two-way clustering.")
    if len(set(clusters)) != len(clusters):
        raise ValueError(
            f"clusters must not repeat a column, got {tuple(clusters)!r} -- "
            "the same dimension clustered on twice would silently reach "
            "statsmodels as if it were two independent clustering "
            "dimensions."
        )
    if np.isinf(frame[[outcome, regressor]].to_numpy(dtype=float)).any():
        raise ValueError(
            f"columns {outcome!r}/{regressor!r} must not contain Infinity."
        )

    wanted = [outcome, regressor, *clusters]
    if fixed_effects is not None:
        wanted.append(fixed_effects)
    columns = list(
        dict.fromkeys(wanted)
    )  # de-duplicate; fixed_effects may equal a cluster
    clean = frame[columns].dropna().copy()
    if clean.empty:
        raise ValueError(
            "No complete (non-missing) rows remain across the required "
            f"columns {columns} -- cannot fit a regression on zero "
            "observations."
        )
    for column in clusters:
        n_unique = clean[column].nunique()
        if n_unique < 2:
            raise ValueError(
                f"cluster column {column!r} has {n_unique} distinct value(s) "
                "after dropping incomplete rows -- at least 2 are required "
                "for cluster-robust standard errors to be well-defined at "
                "all."
            )
    design = pd.DataFrame({regressor: clean[regressor].astype(float)})
    if fixed_effects is not None:
        dummies = pd.get_dummies(
            clean[fixed_effects], prefix=fixed_effects, drop_first=True, dtype=float
        )
        design = pd.concat([design, dummies], axis=1)
    design.insert(0, "const", 1.0)
    # A rank-deficient design (e.g. a constant regressor, or fixed-effects
    # dummies collinear with it) makes OLS's normal equations singular --
    # statsmodels falls back to a pseudo-inverse and returns a fit with no
    # error at all, silently reporting coefficients/SEs for a model that
    # isn't actually identified. Caught here with the specific offending
    # columns rather than left as a plausible-looking but meaningless fit.
    rank = np.linalg.matrix_rank(design.to_numpy(dtype=float))
    if rank < design.shape[1]:
        raise ValueError(
            f"The design matrix (columns {list(design.columns)}) is rank-"
            f"deficient (rank {rank} of {design.shape[1]}) -- likely a "
            "constant regressor or fixed-effects dummies collinear with "
            "it; OLS is not identified on this input."
        )
    groups = np.column_stack([pd.factorize(clean[column])[0] for column in clusters])
    endog = clean[outcome].astype(float).reset_index(drop=True)
    model = sm.OLS(endog, design.reset_index(drop=True))
    cov_kwds: dict[str, Any] = {"groups": groups if len(clusters) > 1 else groups[:, 0]}
    return model.fit(cov_type="cluster", cov_kwds=cov_kwds)


def paired_cluster_bootstrap_sharpe_difference(
    returns_a: pd.Series,
    returns_b: pd.Series,
    cluster_of_date: pd.Series,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> PairedSharpeDifferenceResult:
    """Paired cluster-block bootstrap CI for ``Sharpe(a) - Sharpe(b)``.

    Null hypothesis: the two stitched return series have equal population
    Sharpe ratios. Whole clusters (e.g. walk-forward OOS windows) are
    resampled with replacement for *both* series simultaneously, so the
    pairing (a and b on the same periods) and within-cluster dependence
    are both preserved -- an ordinary two-sample Sharpe-difference test
    would treat every period as independent and ignore that a's and b's
    returns share the same market environment. Works at whatever
    frequency ``returns_a``/``returns_b`` and ``periods_per_year`` are
    consistent with (daily is the common case, matching
    ``periods_per_year``'s own default, but nothing here assumes it).

    ``p_value_two_sided`` is a PERCENTILE bootstrap p-value -- twice the
    smaller one-sided tail share of the bootstrap distribution of the
    OBSERVED (generally nonzero-centered) difference crossing zero, the
    same duality a percentile confidence interval already implies (a 95%
    CI excluding zero corresponds to p < 0.05 here). This is a
    recognized, standard bootstrap testing approach, but a simpler one
    than a genuinely null-recentered or label-permutation test that
    directly enforces "Sharpe(a) == Sharpe(b)" by construction -- treat
    it as a normal bootstrap significance indicator, not a guarantee of
    the exact same finite-sample properties a dedicated permutation test
    would have.

    Args:
        returns_a: Returns of strategy/policy A, one row per period.
        returns_b: Returns of strategy/policy B, same calendar as A.
        cluster_of_date: Cluster label per date (e.g. fold index). Rows
            are aligned by index and then a missing value on any of the
            three drops that date -- the effective date range used is
            the *intersection* of ``returns_a``, ``returns_b`` and
            ``cluster_of_date``, not their union.
        n_boot: Number of cluster-resampled draws.
        seed: Seed for `numpy.random.default_rng`.
        periods_per_year: Annualisation factor for the Sharpe ratio.
        risk_free_rate: Annual risk-free rate subtracted from both series
            before computing each Sharpe ratio (``0.0`` default --
            appropriate when the inputs are already excess returns).
            Converted to a per-period rate via ``risk_free_rate /
            periods_per_year``, the same convention
            ``quantlab.risk.metrics.sharpe_ratio`` uses.

    Returns:
        ``estimate``, ``ci_low``/``ci_high`` (95% percentile interval),
        ``p_value_two_sided`` (see above), ``n_clusters``.
    """
    n_boot = positive_int(n_boot, name="n_boot")
    seed = nonnegative_int(seed, name="seed")
    periods_per_year = positive_int(periods_per_year, name="periods_per_year")
    risk_free_rate = finite_real(risk_free_rate, name="risk_free_rate")
    per_period_rf = risk_free_rate / periods_per_year
    for series, name in (
        (returns_a, "returns_a"),
        (returns_b, "returns_b"),
        (cluster_of_date, "cluster_of_date"),
    ):
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be a pandas Series.")
        if not series.index.is_unique:
            raise ValueError(
                f"{name}'s index must not contain duplicate labels -- a "
                "label-based alignment against a duplicate index produces "
                "a cross-product on that label rather than a clean pairing."
            )
    aligned = pd.concat(
        {"a": returns_a, "b": returns_b, "cluster": cluster_of_date}, axis=1
    ).dropna()
    if aligned.empty:
        raise ValueError(
            "returns_a, returns_b and cluster_of_date share no complete, aligned rows."
        )
    # Same rejection _validate_panel_triplet applies elsewhere in this
    # module: an infinite return would silently propagate into an
    # infinite (or NaN, from inf - inf) Sharpe ratio, which the
    # `np.isfinite` filter on `draws` below would then just as silently
    # drop from the bootstrap distribution with no warning.
    if np.isinf(aligned[["a", "b"]].to_numpy(dtype=float)).any():
        raise ValueError("returns_a and returns_b must not contain Infinity.")
    clusters = aligned["cluster"].to_numpy()
    a = aligned["a"].to_numpy(dtype=float)
    b = aligned["b"].to_numpy(dtype=float)
    unique = _unique_cluster_labels(clusters)
    # A single cluster has no resampling variation to estimate from --
    # every draw would resample that same one cluster with replacement,
    # producing a degenerate zero-width CI that looks confident but
    # reflects no genuine uncertainty, rather than the clear error
    # `_validate_panel_triplet`'s own `min_clusters` enforces for every
    # other function in this module.
    if len(unique) < 2:
        raise ValueError(
            "cluster_of_date must contain at least 2 distinct clusters "
            f"for cluster-level resampling to be well-defined, got {len(unique)}."
        )
    members = {c: np.flatnonzero(clusters == c) for c in unique}
    rng = np.random.default_rng(seed)

    def sharpe(x: np.ndarray) -> float:
        # NaN (never a fabricated 0.0) for degenerate variance (a single
        # observation, or every value identical) -- a bootstrap draw this
        # happens to is uninformative, not evidence of "no difference", and
        # must be *excluded* below rather than silently pulling the whole
        # distribution toward zero.
        if len(x) < 2:
            return float("nan")
        excess = x - per_period_rf
        s = excess.std(ddof=1)
        return (
            float(excess.mean() / s * np.sqrt(periods_per_year))
            if s > 0
            else float("nan")
        )

    observed = sharpe(a) - sharpe(b)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[c] for c in chosen])
        draws[i] = sharpe(a[idx]) - sharpe(b[idx])
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return PairedSharpeDifferenceResult(
            estimate=observed,
            ci_low=float("nan"),
            ci_high=float("nan"),
            p_value_two_sided=float("nan"),
            n_clusters=len(unique),
        )
    # Same finite-sample (count + 1) / (n + 1) correction as
    # within_cluster_permutation_test, so this never reports an
    # impossible exact-zero p-value; explicitly clipped at 1.0 since a
    # draw of exactly 0.0 is counted on both sides and could otherwise
    # push the raw formula fractionally past it.
    below = int((draws <= 0.0).sum())
    above = int((draws >= 0.0).sum())
    p_value = 2.0 * (min(below, above) + 1) / (len(draws) + 1)
    return PairedSharpeDifferenceResult(
        estimate=observed,
        ci_low=float(np.percentile(draws, 2.5)),
        ci_high=float(np.percentile(draws, 97.5)),
        p_value_two_sided=float(min(p_value, 1.0)),
        n_clusters=len(unique),
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control).

    Not a hypothesis test itself: an adjustment applied to a family of
    already-computed p-values so that rejecting every adjusted p-value
    below 0.05 controls the family-wise Type-I error rate at 5%, unlike
    comparing each raw p-value to 0.05 independently.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if len(p) == 0:
        return []
    if np.isnan(p).any():
        raise ValueError("p_values must not contain missing values.")
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("p_values must lie in [0, 1].")
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def cliffs_delta(x: object, y: object) -> float:
    """Cliff's delta: ``P(x > y) - P(x < y)``, a scale-free effect size.

    Not a hypothesis test: a nonparametric measure of how much one
    distribution stochastically dominates the other, in ``[-1, 1]``, with
    0 meaning neither dominates the other on net (``P(x > y) == P(x <
    y)``) -- NOT that the two distributions are otherwise identical or
    indistinguishable: e.g. two distributions with the same median but
    very different spread can still show a delta near 0. Computed via
    the Mann-Whitney U statistic. Returns NaN when either sample has no
    remaining (non-missing) observations after dropping NaN. Infinity is
    NOT silently dropped alongside NaN -- unlike a genuinely missing
    value, it is an ordered, comparable value the Mann-Whitney ranking
    would otherwise silently discard as if it had never existed; it is
    rejected outright instead, the same data-quality stance this module
    takes elsewhere.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")
    if np.isinf(x_arr).any() or np.isinf(y_arr).any():
        raise ValueError("x and y must not contain Infinity.")
    x_arr = x_arr[~np.isnan(x_arr)]
    y_arr = y_arr[~np.isnan(y_arr)]
    if len(x_arr) == 0 or len(y_arr) == 0:
        return float("nan")
    result = stats.mannwhitneyu(x_arr, y_arr, alternative="two-sided")
    u_statistic = float(np.asarray(result[0]).item())
    return 2.0 * u_statistic / (len(x_arr) * len(y_arr)) - 1.0


__all__ = [
    "ClusterDifferenceResult",
    "ClusterSpearmanResult",
    "FamaMacBethResult",
    "PairedSharpeDifferenceResult",
    "PermutationTestResult",
    "Statistic",
    "cliffs_delta",
    "cluster_bootstrap_difference",
    "clustered_regression",
    "fama_macbeth_difference",
    "holm_adjust",
    "paired_cluster_bootstrap_sharpe_difference",
    "per_cluster_spearman",
    "within_cluster_permutation_test",
]
