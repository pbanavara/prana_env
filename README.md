---
title: PRANA-Env Environment Server
emoji: 🏥
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - clinical
---

# PRANA-Env

**Policy Reinforced Administrative Navigation Agent** — an OpenEnv RL environment for kidney transplant administration.

PRANA-Env simulates the multi-step clinical workflow required to file a KARS-compliant SRTR report for a transplant candidate. The agent must query fragmented datastores, detect stale lab values, and file a complete report — earning rewards from a deterministic KARS validator.

## Architecture

```
LLM Agent (GPT-4o / fine-tuned model)
        │
        │  query_db / record_value / file_report
        ▼
  PranaEnv Client  ──(WebSocket)──  PranaEnvironment Server
                                          │
                                    KARS Validator
                                    (reward signal)
```

## Action Space

| Action | Required fields | Effect |
|--------|----------------|--------|
| `query_db` | `target`, `field`, `patient_id` | Returns current value from PatientDB |
| `record_value` | `field`, `value` | Writes value into episode record with today's timestamp |
| `file_report` | — | KARS validates record → reward → done |

## Observation Space

Every observation includes:

```python
PranaObservation(
    query_result      # str: value, NOT_FOUND, RECORDED, KARS status
    active_task       # str: current task context (t1–t5)
    recorded_fields   # dict: {field: {value, recorded_at}} — full current record
    missing_fields    # list[str]: KARS issues after file_report
    kars_result       # str | None: "PASSED" | "FAILED"
    reward            # float
    done              # bool
)
```

`recorded_fields` shows the agent its full current state including timestamps — enabling staleness detection and selective re-querying.

## Reward Signal

| Event | Reward |
|-------|--------|
| KARS PASSED — first attempt | **+15** |
| KARS PASSED — after correction | **+10** |
| Re-query of already-fresh field | **−1** |
| KARS FAILED — missing or stale fields | **−5** |
| KARS FAILED — unrecoverable (3 attempts) | **−10** |

## Temporal Model (T1 → T5)

Episodes simulate a 4-month clinical timeline:

- **T1 (2025-11-07)**: Initial labs recorded. Snapshot pre-loaded into episode record on `reset()`.
- **T5 (2026-03-07)**: Filing date. KARS requires time-sensitive fields within **90 days**.

On `reset()`, the agent sees a pre-populated record with stale T1 values. It must:
1. Identify which fields are stale (`hba1c`, `gfr`, `creatinine` — time-sensitive)
2. Re-query only those fields to get current T5 values
3. Leave stable fields (`blood_type`) untouched — re-querying incurs a penalty
4. File when the record is complete and fresh

**Example trajectory:**
```
reset() → record pre-loaded: {hba1c: {value: 7.2, recorded_at: 2025-11-07}, ...}

query_db(hba1c)      → 8.9   (T5 value — GFR worsened)
query_db(gfr)        → 12.1  (was 18.5 at T1)
query_db(creatinine) → 4.7   (was 3.8 at T1)
record_value × 3
file_report()        → KARS PASSED, reward=+15
```

## Quick Start

```bash
# Start the server
conda activate openenv
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

```python
# Run the LLM agent loop
python test_agent.py
```

```python
# Run N episodes for GRPO rollout batch
from test_agent import run_episodes

trajectories = run_episodes(
    task="File a KARS-compliant SRTR report for patient P001. "
         "A T1 record exists from 4 months ago. "
         "Check which fields are stale, re-query only what's needed, and file.",
    patient_id="P001",
    n=8,  # GRPO batch size
)
```

## Patients

| ID | Condition | T1 GFR | T5 GFR | HbA1c T1→T5 | Notes |
|----|-----------|--------|--------|-------------|-------|
| P001 | CKD Stage 4 | 18.5 | 12.1 | 7.2→8.9 | Complete record |
| P002 | Diabetic nephropathy | 11.0 | 8.3 | 9.1→10.2 | Antihypertensives, insulin |
| P003 | CKD Stage 3 | 22.3 | 19.8 | null | HbA1c never recorded, inactive waitlist |

## KARS Required Fields

| Field | Source | Time-sensitive |
|-------|--------|---------------|
| `hba1c` | PatientDB | Yes — 90-day window |
| `gfr` | PatientDB | Yes — 90-day window |
| `creatinine` | PatientDB | Yes — 90-day window |
| `blood_type` | PatientDB | No — stable |

## Project Structure

```
prana_env/
├── client.py                      # PranaEnv WebSocket client
├── models.py                      # PranaAction, PranaObservation
├── test_agent.py                  # LLM agent RL loop (GPT-4o)
├── test_client.py                 # Smoke test client
├── data/
│   └── patient_db.json            # Patient records with T1 snapshots and T5 values
└── server/
    ├── app.py                     # FastAPI + WebSocket server
    ├── prana_env_environment.py   # RL environment: actions, KARS validator, rewards
    └── Dockerfile
```

## Connecting to an Existing Server

```python
from prana_env.client import PranaEnv
from prana_env.models import PranaAction

with PranaEnv(base_url="http://localhost:8000") as env:
    result = env.reset(patient_id="P001")
    print(result.observation.query_result)

    result = env.step(PranaAction(action_type="query_db", target="PatientDB",
                                  field="hba1c", patient_id="P001"))
    print(result.observation.query_result)   # "8.9"
    print(result.observation.recorded_fields)  # current record state
```

## Deploying to Hugging Face Spaces

```bash
openenv push
# or
openenv push --repo-id my-org/prana-env --private
```

After deployment:
- **Web UI**: `/web`
- **API docs**: `/docs`
- **Health**: `/health`
- **WebSocket**: `/ws`
