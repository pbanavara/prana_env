"""
PRANA-Env Environment Implementation.

Minimal RL loop:
  1. query_db     — retrieve field from PatientDB
  2. record_value — write field into episode patient record
  3. file_report  — KARS validator → reward signal → episode done

Reward:
  +15  KARS PASSED on first attempt
  +10  KARS PASSED after prior failed attempt
   +3  record_value with valid policy_ref (retroactive correction bonus — Phase 2)
   -1  query_db for a field already fresh in the record (inefficiency penalty)
   -5  file_report with missing or stale required fields
  -10  unrecoverable KARS failure (max filing attempts exceeded)

Temporal model (T1 → T5, ~4 months):
  - Episode resets with patient record pre-populated from T1 snapshot (2025-11-07)
  - Current date is EPISODE_DATE (2026-03-07)
  - Time-sensitive fields (hba1c, gfr, creatinine) must be recorded within
    RECENCY_DAYS (90) of EPISODE_DATE to pass KARS
  - Stable fields (blood_type, pra) have no recency requirement
  - T1 snapshot values are 120 days old → stale → agent must re-query and re-record
"""

import logging
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

# KARS required fields for v1 (PatientDB subset)
KARS_REQUIRED_FIELDS = ["hba1c", "gfr", "creatinine", "blood_type"]

# Fields that must be within RECENCY_DAYS of filing date
TIME_SENSITIVE_FIELDS = {"hba1c", "gfr", "creatinine"}

# Stable fields — no recency requirement
STABLE_FIELDS = {"blood_type", "pra"}

# Max file_report attempts before unrecoverable failure
MAX_FILE_ATTEMPTS = 3

# Temporal constants
EPISODE_DATE = date(2026, 3, 7)     # T5 filing date (today)
T1_DATE = date(2025, 11, 7)         # T1 initial labs (~4 months ago)
RECENCY_DAYS = 90                    # KARS recency requirement in days


def kars_validate(record: dict) -> tuple[bool, list[str]]:
    """
    Deterministic KARS validator with recency checks.
    record values are {"value": ..., "recorded_at": "YYYY-MM-DD"}.
    Returns (passed, issues) where issues = missing fields + stale fields.
    """
    cutoff = EPISODE_DATE - timedelta(days=RECENCY_DAYS)
    issues = []

    for f in KARS_REQUIRED_FIELDS:
        entry = record.get(f)
        if entry is None or entry.get("value") is None:
            issues.append(f"{f} (missing)")
            continue
        if f in TIME_SENSITIVE_FIELDS:
            recorded_at_str = entry.get("recorded_at", "")
            try:
                recorded_at = date.fromisoformat(recorded_at_str)
                if recorded_at < cutoff:
                    issues.append(f"{f} (stale: recorded {recorded_at_str}, must be after {cutoff})")
            except ValueError:
                issues.append(f"{f} (invalid date)")

    return (len(issues) == 0, issues)


class PranaEnvironment(Environment):
    """
    PRANA-Env: kidney transplant administration RL environment.

    Episode flow:
      reset()              → load patient, pre-populate T1 stale record
      query_db(field)      → read current (T5) value from PatientDB
      record_value(field)  → write value into episode record with today's date
      file_report()        → KARS validate → reward → done=True on pass

    The agent must detect that T1 values are stale and re-query time-sensitive
    fields (hba1c, gfr, creatinine) before filing.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        logger.info(f"{tag} Initializing PranaEnvironment")
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._active_task = "t1"
        self._patient_id: str | None = None
        self._patient_record: dict = {}
        self._file_attempts: int = 0
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
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._active_task = "t1"
        self._patient_id = patient_id
        self._patient_record = {}
        self._file_attempts = 0

        # Pre-populate record with T1 snapshot (stale — 120 days old)
        if patient_id:
            patients = self._patient_db.get("patients", {})
            patient = patients.get(patient_id, {})
            snapshot = patient.get("t1_snapshot", {})
            t1_recorded_at = date.fromisoformat(snapshot.get("recorded_at", T1_DATE.isoformat()))
            for field in KARS_REQUIRED_FIELDS:
                val = snapshot.get(field)
                if val is not None:
                    self._patient_record[field] = self._make_entry(val, t1_recorded_at)

        cutoff = EPISODE_DATE - timedelta(days=RECENCY_DAYS)
        logger.info(f"{tag} reset episode={self._state.episode_id} patient_id={patient_id} "
                    f"pre_populated={list(self._patient_record.keys())}")

        return PranaObservation(
            query_result=(
                f"Episode reset. Patient: {patient_id or 'not set'}. "
                f"Filing date: {EPISODE_DATE}. "
                f"Required fields: {KARS_REQUIRED_FIELDS}. "
                f"Time-sensitive fields {sorted(TIME_SENSITIVE_FIELDS)} must be recorded after {cutoff}. "
                f"Pre-existing T1 record loaded (recorded {T1_DATE}) — check recency before filing."
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
                    logger.info(f"{tag} field={field} already fresh in record — inefficiency penalty")
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

        value = patient.get(field)
        if value is None:
            return PranaObservation(
                query_result=f"NOT_FOUND: '{field}' has no value for patient '{patient_id}'.",
                active_task=self._active_task,
                recorded_fields=self._patient_record.copy(),
                done=False,
                reward=0.0,
            )

        logger.info(f"{tag} query_db OK field={field} value={value}")
        return PranaObservation(
            query_result=str(value),
            active_task=self._active_task,
            recorded_fields=self._patient_record.copy(),
            done=False,
            reward=0.0,
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
        logger.info(f"{tag} record_value field={field} value={value} record={self._patient_record}")

        required_fresh = sum(
            1 for f in KARS_REQUIRED_FIELDS
            if f in self._patient_record and self._patient_record[f].get("value") is not None
        )
        return PranaObservation(
            query_result=f"RECORDED: {field} = {value} (as of {EPISODE_DATE}). Record has {required_fresh}/{len(KARS_REQUIRED_FIELDS)} required fields.",
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
            f"passed={passed} issues={issues} record={self._patient_record}"
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

        # Unrecoverable failure
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

        # Recoverable failure
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
