"""Thin backend adapter for Koi's authoritative direct AIC surrogate."""

from typing import TYPE_CHECKING

from src.prediction.backends.base import Candidate, SurrogateEstimate

if TYPE_CHECKING:
    from src.prediction.surrogate import SurrogatePrediction


class AICBackend:
    """Delegate unchanged inputs and structured failures to SurrogatePrediction."""

    name = "primary"

    def __init__(self, surrogate: "SurrogatePrediction | None" = None):
        self.surrogate = surrogate
        self.version = (
            "aic:src.prediction.surrogate.SurrogatePrediction"
            if surrogate is None
            else f"aic:{type(surrogate).__module__}.{type(surrogate).__qualname__}"
        )

    def provides(self) -> set[str]:
        return {
            "cost_per_token",
            "p99_ttft_ms",
            "p99_tpot_ms",
            "slo_margin",
            "throughput_token_per_sec",
        }

    def estimate(
        self,
        candidate: Candidate,
        *,
        candidate_graph=None,
        method=("AIC_DynoSim",),
        scenario: str = "mean",
    ) -> SurrogateEstimate:
        surrogate = self.surrogate
        if surrogate is None:
            from src.prediction.surrogate import SurrogatePrediction

            surrogate = SurrogatePrediction()
            self.surrogate = surrogate
        y_hat, v_hat = surrogate.compose_prediction(
            job_config=candidate.job_config,
            job_features=candidate.job_features,
            candidate_graph=candidate_graph,
            method=method,
            scenario=scenario,
        )
        nodes = set(y_hat or {}) | set(v_hat or {})
        return SurrogateEstimate(
            y_hat=dict(y_hat or {}),
            v_hat=dict(v_hat or {}),
            status="success",
            version=self.version,
            coverage=dict.fromkeys(nodes, 1.0),
            source=self.name,
            metadata={"method": list(method) if isinstance(method, list | tuple) else method},
        )
