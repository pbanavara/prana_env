"""
PRANA-Env agent with full minimal RL loop.

The LLM agent must:
  1. query_db      — retrieve required fields from PatientDB
  2. record_value  — write each field into the episode record
  3. file_report   — submit to KARS validator → reward → done

Reward signal:
  +15  KARS PASSED first attempt
  +10  KARS PASSED after correction
   -1  redundant query (field already recorded)
   -5  filed with missing fields (recoverable)
  -10  unrecoverable failure
"""

import json
import openai
from dataclasses import dataclass, field
from typing import Optional
from prana_env.client import PranaEnv
from prana_env.models import PranaAction

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": "Retrieve a specific field from a clinical datastore for a patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target":     {"type": "string", "description": "PatientDB | ClinicalNotesDB | PharmacyDB | WaitlistDB"},
                    "field":      {"type": "string", "description": "Field name (e.g. hba1c, gfr, creatinine, blood_type)"},
                    "patient_id": {"type": "string", "description": "Patient identifier (e.g. P001)"},
                },
                "required": ["target", "field", "patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_value",
            "description": "Write a retrieved field value into the episode patient record.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field":  {"type": "string", "description": "Field name to record"},
                    "value":  {"type": "string", "description": "Value to record"},
                    "source": {"type": "string", "description": "Datastore the value came from"},
                },
                "required": ["field", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_report",
            "description": (
                "Submit the compiled patient record to the KARS validator. "
                "Returns PASSED (done) or FAILED with missing fields. "
                "Call only after recording all required fields: hba1c, gfr, creatinine, blood_type."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

SYSTEM_PROMPT = """You are a kidney transplant administrative agent.

Your goal is to compile a complete patient record and file a KARS-compliant SRTR report.

Required fields: hba1c, gfr, creatinine, blood_type (all from PatientDB).

KARS Recency Policy:
- Time-sensitive fields (hba1c, gfr, creatinine) must be recorded within 90 days of the filing date.
- Stable fields (blood_type) have no recency requirement.
- The episode starts with a pre-existing T1 record (~4 months old). These values are STALE.
- You must re-query and re-record hba1c, gfr, and creatinine before filing.
- Do NOT re-query blood_type — it is stable and already valid.

Workflow:
1. Check recorded_fields in the observation — identify stale time-sensitive fields.
2. Use query_db to retrieve fresh values for stale fields only.
3. Use record_value to write each fresh value into the patient record.
4. Use file_report to submit. If it fails due to stale or missing fields, fix and retry.

Do not guess values. Always query before recording."""

# ── Trajectory dataclass ──────────────────────────────────────────────────────

@dataclass
class Step:
    action: dict
    observation: str
    reward: float
    done: bool

@dataclass
class Trajectory:
    episode_id: str
    steps: list[Step] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    def __repr__(self):
        terminal = next((s for s in reversed(self.steps) if s.done), None)
        kars = terminal.observation if terminal else "incomplete"
        return (
            f"Trajectory(episode={self.episode_id}, "
            f"steps={len(self.steps)}, "
            f"total_reward={self.total_reward}, "
            f"outcome={kars!r})"
        )

# ── RL primitives ─────────────────────────────────────────────────────────────

def reset(env: PranaEnv, patient_id: str) -> str:
    result = env.reset(patient_id=patient_id)
    return result.observation.query_result


def step(env: PranaEnv, action_type: str, **kwargs) -> tuple[str, float, bool, list]:
    result = env.step(PranaAction(action_type=action_type, **kwargs))
    obs    = result.observation
    return (
        obs.query_result,
        obs.reward or 0.0,
        obs.done or False,
        obs.missing_fields or [],
    )


def rollout(env: PranaEnv, task: str, patient_id: str, episode_id: str, max_turns: int = 20) -> Trajectory:
    """Run one full episode. LLM drives the action loop until done=True."""
    llm = openai.OpenAI()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": task},
    ]
    trajectory = Trajectory(episode_id=episode_id)

    print(f"\n── Episode {episode_id} ──────────────────────────────")
    print(f"Task: {task}")

    initial_obs = reset(env, patient_id)
    print(f"[reset] {initial_obs}")

    for turn in range(max_turns):
        response = llm.chat.completions.create(
            model="gpt-4o",
            tools=TOOLS,
            messages=messages,
        )
        msg = response.choices[0].message
        messages.append(msg)

        # No tool calls → LLM finished without filing (shouldn't happen with good prompt)
        if msg.tool_calls is None:
            print(f"[turn {turn+1}] Agent: {msg.content}")
            trajectory.steps.append(Step(
                action={"type": "end_turn"},
                observation=msg.content or "",
                reward=0.0,
                done=True,
            ))
            break

        for tool_call in msg.tool_calls:
            action_type = tool_call.function.name
            inp = json.loads(tool_call.function.arguments)
            print(f"[turn {turn+1}] {action_type}({json.dumps(inp)})")

            obs_str, reward, done, missing = step(env, action_type, **inp)
            print(f"[turn {turn+1}] obs={obs_str!r}  reward={reward}  done={done}")

            trajectory.steps.append(Step(
                action={"type": action_type, **inp},
                observation=obs_str,
                reward=reward,
                done=done,
            ))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": obs_str,
            })

            if done:
                return trajectory

    return trajectory


def run_episodes(task: str, patient_id: str, n: int = 1) -> list[Trajectory]:
    """Run N independent episodes. Set n=8 for GRPO rollout batch."""
    trajectories = []
    with PranaEnv(base_url="http://localhost:8000") as env:
        for i in range(n):
            traj = rollout(env, task, patient_id, episode_id=f"ep_{i+1}")
            trajectories.append(traj)

    print(f"\n── Summary ({n} episode(s)) ──────────────────────────")
    for t in trajectories:
        print(f"  {t}")
    return trajectories


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_episodes(
        task=(
            "File a KARS-compliant SRTR report for patient P001. "
            "A T1 record exists from 4 months ago. "
            "Check which fields are stale, re-query only what's needed, and file."
        ),
        patient_id="P001",
        n=1,  # set n=8 for GRPO rollout batch
    )
