"""
Data models for PRANA-Env.

Action space and observation space for the kidney transplant
administration environment.
"""

from typing import List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class PranaAction(Action):
    """
    Action for PRANA-Env.

    Supported action_types:
      query_db       — retrieve a field from a datastore
      record_value   — write a field into the episode patient record
      file_report    — submit compiled record to KARS validator
    """

    action_type: str = Field(
        ...,
        description=(
            "Type of action: query_db | record_value | file_report"
        ),
    )
    # query_db / record_value
    target: Optional[str] = Field(
        default=None,
        description="Datastore name for query_db (PatientDB, ClinicalNotesDB, PharmacyDB, WaitlistDB)",
    )
    field: Optional[str] = Field(
        default=None, description="Field name to query or record"
    )
    patient_id: Optional[str] = Field(
        default=None, description="Patient identifier"
    )
    # record_value / update_past_record
    value: Optional[str] = Field(
        default=None, description="Value to record"
    )
    source: Optional[str] = Field(
        default=None, description="Source datastore the value was retrieved from"
    )
    task_ref: Optional[str] = Field(
        default=None, description="Task reference for retroactive updates (e.g. 't1')"
    )
    policy_ref: Optional[str] = Field(
        default=None, description="OPTN policy citation (e.g. 'OPTN-18.1.2')"
    )


class PranaObservation(Observation):
    """
    Observation from PRANA-Env.
    """

    query_result: str = Field(
        default="",
        description="Result of the action: field value, NOT_FOUND, or status message",
    )
    active_task: str = Field(
        default="t1",
        description="Current task context (t1-t5)",
    )
    policy_alerts: str = Field(
        default="",
        description="Any OPTN policy rules triggered by this observation",
    )
    # Populated after file_report
    kars_result: Optional[str] = Field(
        default=None,
        description="KARS validation result: PASSED or FAILED",
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Fields missing from the report per KARS requirements",
    )
    recorded_fields: dict = Field(
        default_factory=dict,
        description="Current patient record — fields recorded so far this episode",
    )
