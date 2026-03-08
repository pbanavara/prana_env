"""PRANA-Env Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import PranaAction, PranaObservation


class PranaEnv(EnvClient[PranaAction, PranaObservation, State]):
    """Client for PRANA-Env."""

    def _step_payload(self, action: PranaAction) -> Dict:
        return {k: v for k, v in action.model_dump().items() if v is not None}

    def _parse_result(self, payload: Dict) -> StepResult[PranaObservation]:
        obs_data = payload.get("observation", {})
        observation = PranaObservation(
            query_result=obs_data.get("query_result", ""),
            active_task=obs_data.get("active_task", "t1"),
            policy_alerts=obs_data.get("policy_alerts", ""),
            kars_result=obs_data.get("kars_result"),
            missing_fields=obs_data.get("missing_fields", []),
            recorded_fields=obs_data.get("recorded_fields", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
