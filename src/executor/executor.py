from src.core.models import Plan


class Executor:
    """Abstract base class for plan executors."""

    def send_to_executor(self, plan):
        raise NotImplementedError("Executor subclasses must implement send_to_executor")


class StorePlanExecutor(Executor):
    """Write validated Koi plans to Tandemn Store for Orca to apply."""

    def __init__(self, user_id: str, plan_store=None, postgres_client=None):
        self.user_id = user_id
        self.plan_store = plan_store or self._default_store(postgres_client)

    @staticmethod
    def _default_store(postgres_client=None):
        from tandemn_system_data.clients import (  # type: ignore[import-untyped]
            PlanStore,
            PostgresClient,
        )

        return PlanStore(postgres_client or PostgresClient())

    def send_to_executor(self, plan):
        store_plan = self._to_store_plan(
            plan if isinstance(plan, Plan) else Plan.from_raw(plan, tick=0)
        )
        self.plan_store.create(store_plan)
        return [{"plan_id": store_plan.plan_id, "status": store_plan.status}]

    def _to_store_plan(self, plan: Plan):
        from tandemn_system_data import models as store_models  # type: ignore[import-untyped]

        actions = []
        for action in plan.actions:
            actions.append(
                store_models.PlanAction(
                    job_id=action.job_id,
                    type=store_models.ActionType(action.type.value),
                    ladder=[rank.to_dict() for rank in action.ladder] if action.ladder else None,
                    target_tps=action.target_tps,
                    target_p99_ttft_ms=action.target_p99_ttft_ms,
                    target_p99_tpot_ms=action.target_p99_tpot_ms,
                )
            )
        return store_models.Plan(
            user_id=self.user_id,
            koi_version=plan.koi_version,
            tick_rationale=plan.tick_rationale or "",
            actions=actions,
        )
