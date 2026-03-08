"""
PRANA-Env Environment Implementation.

Minimal RL loop:
  1. query_db     — retrieve field from PatientDB
  2. record_value — write field into episode patient record
  3. file_report  — KARS validator → reward signal → episode done

Reward:
  +15  KARS PASSED on first attempt
  +10  KARS PASSED after prior failed attempt
   -1  query_db for a field already fresh in the record (inefficiency penalty)
   -5  file_report with missing or stale required fields
  -10  unrecoverable KARS failure (max filing attempts exceeded)

Stochasticity (4 sources):
  1. T1 date randomization   — T1 age sampled Uniform(T1_AGE_MIN, T1_AGE_MAX) days
                               Agent must calculate staleness dynamically, not memorize
  2. Random patient selection — if no patient_id given, pick randomly from pool
  3. Anomaly injection        — with ANOMALY_PROB, inject a spurious reading for one
                               time-sensitive field; agent must detect and escalate
  4. Field availability noise — with PENDING_PROB, a field returns PENDING on first
                               query; resolved on retry (simulates data entry lag)
"""

import logging
import random
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4
import json

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import PranaAction, PranaObservation

tag = "[prana_env/environment]"
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# KARS required fields
KARS_REQUIRED_FIELDS = ["hba1c", "gfr", "creatinine", "blood_type"]
TIME_SENSITIVE_FIELDS = {"hba1c", "gfr", "creatinine"}
STABLE_FIELDS = {"blood_type", "pra"}

MAX_FILE_ATTEMPTS = 3

# Temporal constants
EPISODE_DATE = date(2026, 3, 7)
RECENCY_DAYS = 90

# ── Stochasticity parameters ──────────────────────────────────────────────────
T1_AGE_MIN_DAYS = 60       # shortest possible T1 record age (fresh — no re-query needed)
T1_AGE_MAX_DAYS = 150      # longest possible T1 record age (stale — must re-query)
ANOMALY_PROB = 0.30        # probability of injecting anomalous reading per episode
ANOMALY_DELTA = 0.40       # anomalous value deviates by this fraction from true T5
ANOMALY_WINDOW_DAYS = 14   # anomaly detection window (matches OPTN Clinical Integrity Policy)
ANOMALY_THRESHOLD = 0.25   # flag if delta > 25% within window
PENDING_PROB = 0.15        # probability of PENDING response on first query of a field


def kars_validate(record: dict) -> tuple[bool, list[str]]:
    """
    Deterministic KARS validator with recency checks.
    record values: {field: {"value": ..., "recorded_at": "YYYY-MM-DD"}}
    Returns (passed, issues).
    """
    cutoff = EPISODE_DATE - timedelta(days=RECENCY_DAYS)
    issues = []

    for f in KARS_REQUIRED_FIELDS:
        entry = record.get(f)
        if entry is None or entry.get("value") is None:
            issues.append(f"{f} (missing)")
            continue
        if f in TIME_SENSITIVE_FIELDS:
            try:
                recorded_at = date.fromisoformat(entry.get("recorded_at", ""))
                if recorded_at < cutoff:
                    issues.append(f"{f} (stale: recorded {recorded_at}, must be after {cutoff})")
            except ValueError:
                issues.append(f"{f} (invalid date)")

    return (len(issues) == 0, issues)


class PranaEnvironment(Environment):
    """
    PRANA-Env: kidney transplant administration RL environment.

    Stochastic per-episode:
      - T1 record age varies (60–150 days) — agent must calculate recency dynamically
      - Patient selected randomly if not specified
      - One time-sensitive field may have an injected anomalous reading (30% episodes)
      - Some fields return PENDING on first query (15% per field) — retry resolves
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        logger.info(f"{tag} Initializing PranaEnvironment")
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._active_task = "t1"
        self._patient_id: str | None = None
        self._patient_record: dict = {}
        self._file_attempts: int = 0
        self._t1_date: date = EPISODE_DATE - timedelta(days=120)
        self._pending_fields: set = set()
        self._injected_anomaly: dict | None = None
        self._patient_db = self._load_db("patient_db.json")
        logger.info(f"{tag} Loaded PatientDB with {len(self._patient_db.get('patients', {}))} patients")

    def _load_db(self, filename: str) -> dict:
        path = DATA_DIR / filename
        with open(path) as f:
            return json.load(f)

    def _make_entry(self, value, recorded_at: date) -> dict:
        return {"value": str(value), "recorded_at": recorded_at.isoformat()}

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> PranaObservation:
        patient_id: str | None = kwargs.get("patient_id")
        patients = self._patient_db.get("patients", {})

        # ── Stochasticity 2: random patient selection ─────────────────────────
        if not patient_id:
            patient_id = random.choice(list(patients.keys()))
            logger.info(f"{tag} No patient_id specified — randomly selected {patient_id}")

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._active_task = "t1"
        self._patient_id = patient_id
        self._patient_record = {}
        self._file_attempts = 0
        self._pending_fields = set()
        self._injected_anomaly = None

        # ── Stochasticity 1: randomize T1 record age ──────────────────────────
        t1_days_ago = random.randint(T1_AGE_MIN_DAYS, T1_AGE_MAX_DAYS)
        self._t1_date = EPISODE_DATE - timedelta(days=t1_days_ago)
        cutoff = EPISODE_DATE - timedelta(days=RECENCY_DAYS)
        t1_is_stale = self._t1_date < cutoff

        # Pre-populate record with T1 snapshot at randomized date
        patient = patients.get(patient_id, {})
        snapshot = patient.get("t1_snapshot", {})
        for field in KARS_REQUIRED_FIELDS:
            val = snapshot.get(field)
            if val is not None:
                self._patient_record[field] = self._make_entry(val, self._t1_date)

        # ── Stochasticity 3: anomaly injection ────────────────────────────────
        if random.random() < ANOMALY_PROB:
            field = random.choice(sorted(TIME_SENSITIVE_FIELDS))
            t5_value = patient.get(field)
            if t5_value is not None:
                direction = random.choice([-1, 1])
                anomaly_value = round(t5_value * (1 + direction * ANOMALY_DELTA), 1)
                anomaly_days = random.randint(1, 6)
                self._injected_anomaly = {
                    "field": field,
                    "value": anomaly_value,
                    "recorded_at": (EPISODE_DATE - timedelta(days=anomaly_days)).isoformat(),
                }
                logger.info(f"{tag} Injected anomaly: {self._injected_anomaly}")

        logger.info(
            f"{tag} reset episode={self._state.episode_id} patient={patient_id} "
            f"t1_date={self._t1_date} t1_stale={t1_is_stale} "
            f"anomaly={self._injected_anomaly}"
        )

        stale_note = (
            f"T1 record is {'STALE (>90 days)' if t1_is_stale else 'FRESH (≤90 days)'}."
        )

        return PranaObservation(
            query_result=(
                f"Episode reset. Patient: {patient_id}. "
                f"Filing date: {EPISODE_DATE}. "
                f"T1 record date: {self._t1_date} ({t1_days_ago} days ago). {stale_note} "
                f"Required fields: {KARS_REQUIRED_FIELDS}. "
                f"Time-sensitive {sorted(TIME_SENSITIVE_FIELDS)} must be recorded after {cutoff}."
            ),
            active_task=self._active_task,
            recorded_fields=self._patient_record.copy(),
            done=False,
            reward=0.0,
        )

    def step(self, action: PranaAction) -> PranaObservation:  # type: ignore[override]
        self._state.step_count += 1
        logger.info(
            f"{tag} step={self._state.step_count} action_type={action.action_type} "
            f"field={action.field} value={action.value}"
        )

        if action.action_type == "query_db":
            return self._handle_query_db(action)
        if action.action_type == "record_value":
            return self._handle_record_value(action)
        if action.action_type == "file_report":
            return self._handle_file_report(action)

        logger.warning(f"{tag} Unsupported action_type={action.action_type}")
        return PranaObservation(
            query_result=f"NOT_SUPPORTED: action_type '{action.action_type}'.",
            active_task=self._active_task,
            recorded_fields=self._patient_record.copy(),
            done=False,
            reward=0.0,
        )

    # ── Action handlers ───────────────────────────────────────────────────────

    def _handle_query_db(self, action: PranaAction) -> PranaObservation:
        db_name = (action.target or "").lower()
        field = (action.field or "").lower()
        patient_id = action.patient_id or self._patient_id

        if db_name != "patientdb":
            return PranaObservation(
                query_result=f"NOT_AVAILABLE: datastore '{action.target}' not in Phase 1.",
                active_task=self._active_task,
                recorded_fields=self._patient_record.copy(),
                done=False,
                reward=0.0,
            )

        if not patient_id:
            return PranaObservation(
                query_result="ERROR: patient_id required.",
                active_task=self._active_task,
                recorded_fields=self._patient_record.copy(),
                done=False,
                reward=0.0,
            )

        # Inefficiency penalty — field already fresh in record
        cutoff = EPISODE_DATE - timedelta(days=RECENCY_DAYS)
        if field in self._patient_record:
            entry = self._patient_record[field]
            try:
                recorded_at = date.fromisoformat(entry.get("recorded_at", ""))
                if field in STABLE_FIELDS or recorded_at >= cutoff:
                    logger.info(f"{tag} field={field} already fresh — inefficiency penalty")
                    return PranaObservation(
                        query_result=f"ALREADY_RECORDED: '{field}' = {entry['value']} (recorded {entry['recorded_at']})",
                        active_task=self._active_task,
                        recorded_fields=self._patient_record.copy(),
                        done=False,
                        reward=-1.0,
                    )
            except ValueError:
                pass

        patients = self._patient_db.get("patients", {})
        patient = patients.get(patient_id)
        if not patient:
            return PranaObservation(
                query_result=f"NOT_FOUND: patient '{patient_id}' not in PatientDB.",
                active_task=self._active_task,
                recorded_fields=self._patient_record.copy(),
                done=False,
                reward=0.0,
            )

        # ── Stochasticity 4: field availability noise (PENDING) ───────────────
        if field in TIME_SENSITIVE_FIELDS and field not in self._pending_fields:
            if random.random() < PENDING_PROB:
                self._pending_fields.add(field)
                logger.info(f"{tag} field={field} returned PENDING (will resolve on retry)")
                return PranaObservation(
                    query_result=(
                        f"PENDING: '{field}' not yet entered for patient '{patient_id}'. "
                        f"Data entry in progress — retry."
                    ),
                    active_task=self._active_task,
                    recorded_fields=self._patient_record.copy(),
                    done=False,
                    reward=0.0,
                )

        value = patient.get(field)
        if value is None:
            return PranaObservation(
                query_result=f"NOT_FOUND: '{field}' has no value for patient '{patient_id}'.",
                active_task=self._active_task,
                recorded_fields=self._patient_record.copy(),
                done=False,
                reward=0.0,
            )

        # ── Stochasticity 3: include anomaly in history if injected ───────────
        if field in TIME_SENSITIVE_FIELDS:
            query_result = self._format_lab_history(field, patient_id, value)
        else:
            query_result = str(value)

        logger.info(f"{tag} query_db OK field={field} value={value}")
        return PranaObservation(
            query_result=query_result,
            active_task=self._active_task,
            recorded_fields=self._patient_record.copy(),
            done=False,
            reward=0.0,
        )

    def _format_lab_history(self, field: str, patient_id: str, t5_value) -> str:
        """
        Format a time-sensitive field as a timestamped history.
        Includes T1 snapshot entry, T5 current entry, and injected anomaly if present.
        Flags anomalies per OPTN Clinical Integrity Policy.
        """
        snapshot = self._patient_db["patients"][patient_id].get("t1_snapshot", {})
        t1_val = snapshot.get(field)

        history: list[tuple[date, float]] = []
        if t1_val is not None:
            history.append((self._t1_date, float(t1_val)))

        # Inject anomalous reading if this is the affected field
        if self._injected_anomaly and self._injected_anomaly["field"] == field:
            anom_date = date.fromisoformat(self._injected_anomaly["recorded_at"])
            history.append((anom_date, self._injected_anomaly["value"]))

        history.append((EPISODE_DATE, float(t5_value)))

        # Shuffle deterministically by (patient_id, field) — agent must sort by date.
        # No ← latest pointer, no anomaly flag — matches tau2 benchmark behaviour.
        rng = random.Random(hash((patient_id, field)) & 0xFFFFFFFF)
        rng.shuffle(history)

        lines = [f"  {v} (recorded: {d})" for d, v in history]

        return (
            f"{field} measurement history for {patient_id} "
            f"(filing date: {EPISODE_DATE}):\n" + "\n".join(lines)
        )

    def _handle_record_value(self, action: PranaAction) -> PranaObservation:
        field = (action.field or "").lower()
        value = action.value

        if not field or value is None:
            return PranaObservation(
                query_result="ERROR: field and value are required for record_value.",
                active_task=self._active_task,
                recorded_fields=self._patient_record.copy(),
                done=False,
                reward=0.0,
            )

        self._patient_record[field] = self._make_entry(value, EPISODE_DATE)
        logger.info(f"{tag} record_value field={field} value={value}")

        required_fresh = sum(
            1 for f in KARS_REQUIRED_FIELDS
            if f in self._patient_record and self._patient_record[f].get("value") is not None
        )
        return PranaObservation(
            query_result=(
                f"RECORDED: {field} = {value} (as of {EPISODE_DATE}). "
                f"Record has {required_fresh}/{len(KARS_REQUIRED_FIELDS)} required fields."
            ),
            active_task=self._active_task,
            recorded_fields=self._patient_record.copy(),
            done=False,
            reward=0.0,
        )

    def _handle_file_report(self, action: PranaAction) -> PranaObservation:
        self._file_attempts += 1
        passed, issues = kars_validate(self._patient_record)

        logger.info(
            f"{tag} file_report attempt={self._file_attempts} "
            f"passed={passed} issues={issues}"
        )

        if passed:
            reward = 15.0 if self._file_attempts == 1 else 10.0
            logger.info(f"{tag} KARS PASSED reward={reward}")
            return PranaObservation(
                query_result="KARS PASSED. SRTR report accepted.",
                active_task=self._active_task,
                kars_result="PASSED",
                missing_fields=[],
                recorded_fields=self._patient_record.copy(),
                done=True,
                reward=reward,
            )

        if self._file_attempts >= MAX_FILE_ATTEMPTS:
            logger.warning(f"{tag} KARS FAILED unrecoverable after {self._file_attempts} attempts")
            return PranaObservation(
                query_result=f"KARS FAILED (unrecoverable). Issues: {issues}",
                active_task=self._active_task,
                kars_result="FAILED",
                missing_fields=issues,
                recorded_fields=self._patient_record.copy(),
                done=True,
                reward=-10.0,
            )

        logger.info(f"{tag} KARS FAILED recoverable issues={issues}")
        return PranaObservation(
            query_result=f"KARS FAILED. Issues: {issues}. Fix and file again.",
            active_task=self._active_task,
            kars_result="FAILED",
            missing_fields=issues,
            recorded_fields=self._patient_record.copy(),
            done=False,
            reward=-5.0,
        )

    @property
    def state(self) -> State:
        return self._state
