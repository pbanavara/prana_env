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

**Policy Reinforced Administrative Navigation Agent** — an RL environment for clinical administrative workflow agents, applied to kidney transplant candidate management.

Lifelong treatments such as organ transplants involve inherently long-horizon administrative workflows: labs expire, records fragment across systems, regulatory deadlines are unforgiving, and clinical anomalies must be caught before filing. PRANA-Env models this complexity as a reinforcement learning problem, providing a stochastic, multi-scenario environment where an agent learns to navigate real-world clinical constraints rather than memorize fixed trajectories.

The agent must query fragmented clinical datastores, reason about temporal validity of lab results, detect anomalous measurements per OPTN policy, and file a complete KARS-compliant SRTR report — receiving reward from a deterministic validator that mirrors actual regulatory requirements.

A companion **tau2 benchmark** measures agent performance on temporal reasoning and anomaly detection tasks before and after fine-tuning, enabling rigorous evaluation of improvement.

## Why This Is Hard

- **Long horizon**: A single filing episode spans a 4-month clinical timeline (T1 labs → T5 filing date)
- **Stale data**: Time-sensitive fields (HbA1c, GFR, creatinine) expire after 90 days; stable fields (blood type) do not — the agent must distinguish
- **Missing data**: Not all patients have all labs. Non-diabetic patients typically lack HbA1c entirely
- **Anomaly detection**: Two measurements of the same field within 14 days differing by >25% must be flagged — filing is blocked until resolved
- **Distractor fields**: The datastore contains queryable but non-required fields (cholesterol, BMI, albumin, hemoglobin) — the agent must not waste steps on them
- **Stochastic episodes**: T1 record age varies per episode, anomalies are injected randomly, and field availability noise simulates real data-entry lag

## Architecture

```
LLM Agent (GPT-4o / fine-tuned Qwen3-8B)
        │
        │  query_db / record_value / file_report
        ▼
  PranaEnv Client  ──(WebSocket)──  PranaEnvironment Server
                                          │
                                    KARS Validator
                                    (deterministic reward signal)
```

## Action Space

| Action | Required fields | Effect |
|--------|----------------|--------|
| `query_db` | `target`, `field`, `patient_id` | Retrieves field history from PatientDB |
| `record_value` | `field`, `value` | Writes value into episode record with today's timestamp |
| `file_report` | — | KARS validates record → reward → episode done |

## Observation Space

Every step returns:

```python
PranaObservation(
    query_result      # str: field history, NOT_FOUND, RECORDED, or KARS status
    active_task       # str: current task context
    recorded_fields   # dict: {field: {value, recorded_at}} — full current record state
    missing_fields    # list[str]: KARS issues reported after file_report
    kars_result       # str | None: "PASSED" | "FAILED"
    reward            # float
    done              # bool
)
```

`recorded_fields` gives the agent full visibility into its accumulated record including timestamps, enabling it to reason about which fields are fresh and which need re-querying.

## Reward Signal

| Event | Reward |
|-------|--------|
| KARS PASSED — first attempt | **+15** |
| KARS PASSED — after correction | **+10** |
| Re-query of already-fresh field | **−1** |
| KARS FAILED — missing or stale fields | **−5** |
| KARS FAILED — unrecoverable (3 attempts) | **−10** |

## Temporal Model

Episodes simulate a 4-month kidney transplant candidacy timeline:

- **T1**: Initial labs recorded (~4 months before filing). Snapshot pre-loaded into the episode record on `reset()`. T1 age is randomized per episode (60–150 days) — the agent cannot assume staleness; it must calculate it.
- **T5 (filing date: 2026-03-07)**: KARS requires time-sensitive labs within **90 days** of filing.

On `reset()`, the agent receives a pre-populated record with T1 values. It must:
1. Identify which fields are stale (`hba1c`, `gfr`, `creatinine`)
2. Re-query only those fields to retrieve current T5 values
3. Leave stable fields (`blood_type`) untouched — unnecessary re-queries incur a penalty
4. Detect anomalies before filing — if two readings of the same field within 14 days differ by >25%, escalate rather than file

**Example trajectory (no anomaly):**
```
reset() → record pre-loaded with T1 snapshot (stale)

query_db(hba1c)      → history: [7.2 @ 2025-11-07, 8.9 @ 2026-03-01]
query_db(gfr)        → history: [18.5 @ 2025-11-07, 12.1 @ 2026-03-01]
query_db(creatinine) → history: [3.8 @ 2025-11-07, 4.7 @ 2026-03-01]
record_value(hba1c, 8.9)
record_value(gfr, 12.1)
record_value(creatinine, 4.7)
file_report()        → KARS PASSED, reward = +15
```

**Example trajectory (anomaly detected):**
```
query_db(gfr) → history: [18.5 @ 2025-11-07, 6.6 @ 2026-03-01, 12.1 @ 2026-03-05]
                          ↑ 45% drop within 4 days — OPTN integrity policy triggered
→ Do NOT file. Communicate anomaly, recommend confirmatory test.
```

## Patient Database

50 procedurally generated kidney transplant candidates (P001–P050) across CKD stages 3–5:

| ID | Condition | Notes |
|----|-----------|-------|
| P001 | CKD Stage 4 | Complete record |
| P002 | Diabetic nephropathy | HbA1c elevated and worsening |
| P003 | CKD Stage 3 | Non-diabetic — HbA1c absent |
| P004–P050 | CKD Stage 3/4/5 | Procedurally generated (seed=42) |

**Distribution:**
- Stage 3: ~25% · Stage 4: ~50% · Stage 5: ~25%
- 60% diabetic (HbA1c present); non-diabetic patients have an 85% chance of missing HbA1c entirely
- ~10% of patients carry an injected anomalous lab reading for benchmark coverage

**Distractor fields** (queryable, not KARS-required): `cholesterol`, `bmi`, `albumin`, `hemoglobin`

```bash
# Regenerate patient database
python data/generate_patients.py
```

## KARS Required Fields

| Field | Time-sensitive | Rule |
|-------|---------------|------|
| `hba1c` | Yes | Must be recorded within 90 days of filing |
| `gfr` | Yes | Must be recorded within 90 days of filing |
| `creatinine` | Yes | Must be recorded within 90 days of filing |
| `blood_type` | No | Stable — no recency requirement |

## tau2 Benchmark

PRANA-Env ships with a [tau2](https://github.com/sierra-research/tau2-bench) benchmark suite that evaluates agent performance on:

- **Temporal reasoning**: Correctly identifying stale vs. fresh labs across stochastic T1 ages
- **Anomaly detection**: Catching >25% measurement deltas within 14 days per OPTN policy
- **Distractor filtering**: Filing without querying non-required fields

Baseline (Qwen3-8B, untuned): **0.71 Pass@1** on temporal/anomaly tasks.
Target after GRPO fine-tuning: **≥ 0.90 Pass@1**.

Fine-tuning notebook: `prana_grpo_qwen3_8b_fp8.ipynb` (Qwen3-8B FP8, H100 required).

## Quick Start

```bash
# Start the server
conda activate openenv
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

```python
# Connect and run an episode
from prana_env.client import PranaEnv
from prana_env.models import PranaAction

with PranaEnv(base_url="http://localhost:7860") as env:
    obs = env.reset(patient_id="P001")
    print(obs.observation.query_result)

    result = env.step(PranaAction(
        action_type="query_db",
        target="PatientDB",
        field="hba1c",
        patient_id="P001",
    ))
    print(result.observation.query_result)    # lab history with timestamps
    print(result.observation.recorded_fields) # current record state
```

## Project Structure

```
prana_env/
├── client.py                      # PranaEnv WebSocket client
├── models.py                      # PranaAction, PranaObservation
├── test_agent.py                  # LLM agent loop
├── prana_grpo_qwen3_8b_fp8.ipynb  # GRPO fine-tuning notebook (Qwen3-8B FP8, H100)
├── data/
│   ├── patient_db.json            # 50 patients: T1 snapshots, T5 values, distractor fields
│   └── generate_patients.py      # Procedural patient generator (CKD stage distributions, seed=42)
└── server/
    ├── app.py                     # FastAPI + WebSocket server (port 7860)
    ├── prana_env_environment.py   # RL environment: actions, KARS validator, stochasticity
    └── Dockerfile
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
