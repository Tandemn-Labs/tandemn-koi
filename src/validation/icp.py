"""
ICP tests whether an edge is stable across environments.

A causal edge should have invariant residuals across market, cloud, region, zone, or GPU.
A spurious edge often shows env-specific residual shifts.

    residual = observed_value - surrogate_predicted_value

The surrogate removes expected config-driven variation, so ICP checks whether the
remaining error is environment-driven.

Default test: F-test / ANOVA over residuals by env.
Fallback: permutation test with shuffled env labels.

Result:
    p < alpha       -> reject invariance
    p > 1 - alpha   -> accept invariance
    else            -> undecided

If there are too few envs or samples per env, return undecided.
"""

from collections import defaultdict
from collections.abc import Hashable
from enum import Enum

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


class ICPResult(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNDECIDED = "undecided"


class ICP:
    def compute_icp_per_edge(
        self,
        edge,
        evidence_store,
        alpha_icp: float = 0.05,
        min_envs: int = 3,
        n_b: int = 15,
        test: str = "f_test",
        n_permutations: int = 1000,
    ) -> ICPResult:
        """
        Definition: Test whether the surrogate's residual for edge.dst is
                    invariant across environments. Uses pre-computed
                    anchor-regression residuals from EvidenceStore
                    (observed - surrogate_predicted, logged at deploy time)
                    and the F-test by default.
        Usage:      Validator.s2_validate calls once per edge in every
                    deployed mechanism's bundle. The (Quadrant, ICPResult)
                    pair drives delta_c(e) updates via the lookup table.
        Inputs:
            edge           : Edge (.edge_id, .src, .dst)
            evidence_store : EvidenceStore exposing get_rows_for_edge(id);
                             each row has .env_label plus residuals_per_v /
                             residuals_per_y keyed by observed variable.
            alpha_icp      : significance level (default 0.05)
            min_envs       : minimum distinct envs required (default 3)
            n_b            : minimum samples per env for power (default 15)
            test           : "f_test" (default) or "permutation"
            n_permutations : permutation-test trials (only if test="permutation")
        Outputs: ICPResult
        Notes:
            No surrogate call here; residuals are pre-computed and stored
            at deploy time. This function just groups them by env and runs
            the chosen statistical test.
        """
        rows = evidence_store.get_rows_for_edge(edge.edge_id, limit=None)
        if not rows:
            return ICPResult.UNDECIDED

        by_env: dict[Hashable, list[float]] = defaultdict(list)
        for r in rows:
            # EvidenceRow stores residual series by destination variable;
            # an edge's ICP residual is the residual of edge.dst.
            residuals_by_name = r.residuals_per_v if edge.dst_type == "V" else r.residuals_per_y
            residuals = residuals_by_name.get(edge.dst)
            if residuals is None:
                continue
            by_env[r.env_label].extend(np.asarray(residuals, dtype=float).ravel())

        envs_with_power = [e for e, lst in by_env.items() if len(lst) >= n_b]
        if len(envs_with_power) < min_envs:
            return ICPResult.UNDECIDED

        residuals_by_env = {e: np.array(by_env[e], dtype=float) for e in envs_with_power}

        if test == "f_test":
            p_value = self._f_test_invariance(residuals_by_env)
        elif test == "permutation":
            p_value = self._permutation_test_invariance(residuals_by_env, n_permutations)
        else:
            raise ValueError(f"Unknown test backend: {test!r}")

        if p_value < alpha_icp:
            return ICPResult.REJECT
        if p_value > 1.0 - alpha_icp:
            return ICPResult.ACCEPT
        return ICPResult.UNDECIDED

    def compute_icp_for_mechanism(
        self,
        mechanism_id,
        candidate_graph,
        mechanism_registry,
        evidence_store,
        alpha_icp: float = 0.05,
        min_envs: int = 3,
        n_b: int = 15,
        test: str = "f_test",
        n_permutations: int = 1000,
    ) -> dict[str, ICPResult]:
        """
        I think this might not be needed since we are not using ICP for mechanisms.
        """
        mechanism = mechanism_registry.get_mechanism(mechanism_id)
        results = {}
        for edge_id in mechanism.edge_ids:
            edge = candidate_graph.edge_table[edge_id]
            results[edge_id] = self.compute_icp_per_edge(
                edge=edge,
                evidence_store=evidence_store,
                alpha_icp=alpha_icp,
                min_envs=min_envs,
                n_b=n_b,
                test=test,
                n_permutations=n_permutations,
            )
        return results

    def _f_test_invariance(
        self,
        residuals_by_env: dict[Hashable, np.ndarray],
    ) -> float:
        """
        Definition: One-way ANOVA F-test on residuals grouped by env.
                        H_0: residuals identically distributed across envs.
                        F  = MSB / MSW
                        p  = 1 - F_cdf(F; df1=k-1, df2=N-k)
        Usage:      Default test backend for compute_icp_per_edge.
        Inputs:
            residuals_by_env : env_label -> np.ndarray of residuals
        Outputs: p_value : float in [0, 1]
        """
        groups = list(residuals_by_env.values())
        if any(g.size < 2 for g in groups):
            return 0.5
        try:
            _f_stat, p_value = stats.f_oneway(*groups)
        except Exception:
            return 0.5
        if p_value is None or np.isnan(p_value):
            return 0.5
        return float(p_value)

    def _permutation_test_invariance(
        self,
        residuals_by_env: dict[Hashable, np.ndarray],
        n_perm: int,
    ) -> float:
        """
        Definition: Non-parametric invariance test.
                    Statistic: max ratio of per-env variances.
                    Shuffle env labels n_perm times; p-value with
                    add-one smoothing.
                        p = (1 + #{stat_perm >= stat_obs}) / (n_perm + 1)
        Usage:      Slower alternative to F-test when residuals are
                    far from normal.
        Inputs:
            residuals_by_env : env_label -> np.ndarray
            n_perm           : permutation count
        Outputs: p_value : float in (0, 1]
        """
        envs = list(residuals_by_env.keys())
        observed_stat = self._cross_env_variance_ratio(residuals_by_env)

        all_residuals = np.concatenate([residuals_by_env[e] for e in envs])
        env_sizes = [len(residuals_by_env[e]) for e in envs]
        rng = np.random.default_rng()

        count_geq = 0
        for _ in range(n_perm):
            permuted = rng.permutation(all_residuals)
            shuffled: dict[Hashable, np.ndarray] = {}
            idx = 0
            for env, k in zip(envs, env_sizes, strict=True):
                shuffled[env] = permuted[idx : idx + k]
                idx += k
            if self._cross_env_variance_ratio(shuffled) >= observed_stat:
                count_geq += 1

        return (count_geq + 1) / (n_perm + 1)

    @staticmethod
    def _cross_env_variance_ratio(
        residuals_by_env: dict[Hashable, np.ndarray],
    ) -> float:
        """
        Definition: max(var) / min(var) across environments, Larger = stronger evidence
                    against invariance.
        Usage:      Inner statistic for the permutation test.
        Inputs:
            residuals_by_env : env_label -> np.ndarray
        Outputs: float >= 1
        Notes:
            +1e-9 added per variance for numerical stability with zero-variance groups.
        """
        variances = [float(np.var(r)) + 1e-9 for r in residuals_by_env.values()]
        if not variances:
            return 1.0
        return max(variances) / min(variances)
