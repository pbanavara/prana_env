#!/usr/bin/env python3
"""
Generate procedural patient database for PRANA-Env and tau2 benchmark.

Produces 50 patients (P001-P050) across CKD stages 3-5 with:
  - Stage-appropriate GFR / creatinine distributions
  - Diabetic status driving HbA1c presence
  - Systematic missing fields (non-diabetic → null HbA1c, etc.)
  - Distractor fields (cholesterol, bmi, albumin, hemoglobin)
  - T1 snapshot with slightly better values (disease progression)
  - Anomaly injection for ~10% of patients (for benchmark coverage)

Outputs:
  - prana_env/data/patient_db.json        (prana_env flat format)
  - tau2-bench/data/tau2/domains/prana/db.json  (tau2 LabResult format, preserves other DBs)
"""

import json
import random
from pathlib import Path
from datetime import date, timedelta

SEED = 42
N_PATIENTS = 50
EPISODE_DATE = date(2026, 3, 7)
T1_NOMINAL_DATE = date(2025, 11, 7)
T5_MEAS_DATE = date(2026, 3, 1)      # recent measurement date used in tau2 history

rng = random.Random(SEED)

# ── Clinical distributions ─────────────────────────────────────────────────────

CKD_STAGES = {
    3: {"gfr": (30, 59), "creatinine": (1.2, 2.5), "weight": 0.25},
    4: {"gfr": (15, 29), "creatinine": (2.5, 5.0), "weight": 0.50},
    5: {"gfr": (5,  14), "creatinine": (5.0, 9.5), "weight": 0.25},
}

BLOOD_TYPES  = ["O+",  "A+",  "B+",  "AB+", "O-",  "A-",  "B-",  "AB-"]
BLOOD_WGTS   = [0.38, 0.34,  0.09,  0.03,  0.07,  0.06,  0.02,  0.01]

FIRST_NAMES = ["James","Maria","David","Sarah","Michael","Linda","Robert","Patricia",
               "William","Barbara","Richard","Susan","Joseph","Jessica","Thomas","Karen",
               "Charles","Lisa","Christopher","Nancy","Daniel","Betty","Matthew","Margaret",
               "Anthony","Sandra","Mark","Ashley","Donald","Dorothy","Steven","Kimberly",
               "Paul","Emily","Andrew","Donna","Joshua","Michelle","Kenneth","Carol",
               "Kevin","Amanda","Brian","Melissa","George","Deborah","Timothy","Stephanie",
               "Ronald","Rebecca"]
LAST_NAMES  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
               "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
               "Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson",
               "White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker",
               "Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
               "Green","Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell",
               "Carter","Roberts"]

# ── Anomaly patients — fixed set for benchmark reproducibility ─────────────────
# These patients have an extra measurement close to filing that triggers >25% delta
ANOMALY_PATIENT_INDICES = {7, 12, 19, 26, 33}   # 0-indexed within P004-P050


def pick_ckd_stage() -> int:
    stages = list(CKD_STAGES.keys())
    weights = [CKD_STAGES[s]["weight"] for s in stages]
    return rng.choices(stages, weights=weights)[0]


def generate_patient(idx: int) -> dict:
    """Return a patient dict in prana_env format."""
    patient_id = f"P{idx:03d}"
    stage = pick_ckd_stage()
    cfg = CKD_STAGES[stage]

    # T5 current values
    gfr_t5 = round(rng.uniform(*cfg["gfr"]), 1)
    creatinine_t5 = round(rng.uniform(*cfg["creatinine"]), 1)

    diabetic = rng.random() < 0.60
    hba1c_t5 = round(rng.uniform(6.5, 12.0), 1) if diabetic else None

    # Missing field scenarios
    # Non-diabetic patients: 85% chance HbA1c not measured
    missing_hba1c = (not diabetic and rng.random() < 0.85) or (diabetic and rng.random() < 0.04)
    missing_creatinine = rng.random() < 0.05
    missing_blood_type = rng.random() < 0.03

    blood_type = rng.choices(BLOOD_TYPES, weights=BLOOD_WGTS)[0] if not missing_blood_type else None
    pra = round(rng.uniform(0, 80), 1)

    # T1 values: disease was less advanced — GFR higher, creatinine lower
    gfr_t1 = round(min(gfr_t5 * rng.uniform(1.15, 1.60), 60.0), 1)
    creatinine_t1 = round(creatinine_t5 * rng.uniform(0.55, 0.85), 1)
    hba1c_t1 = round(hba1c_t5 * rng.uniform(0.82, 0.96), 1) if hba1c_t5 is not None else None

    # Distractor fields — present, queryable, not KARS-required
    cholesterol = round(rng.uniform(140, 270), 1)
    bmi         = round(rng.uniform(18.5, 40.0), 1)
    albumin     = round(rng.uniform(2.0, 4.2), 2)   # Low in CKD
    hemoglobin  = round(rng.uniform(7.5, 13.5), 1)  # Anemia common in CKD

    t1_snapshot: dict = {
        "gfr": gfr_t1,
        "creatinine": creatinine_t1 if not missing_creatinine else None,
        "blood_type": blood_type,
        "pra": pra,
        "recorded_at": T1_NOMINAL_DATE.isoformat(),
    }
    if hba1c_t1 is not None and not missing_hba1c:
        t1_snapshot["hba1c"] = hba1c_t1

    patient: dict = {
        "patient_id": patient_id,
        "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
        "age": rng.randint(28, 72),
        "ckd_stage": stage,
        "gfr": gfr_t5,
        "creatinine": creatinine_t5 if not missing_creatinine else None,
        "blood_type": blood_type,
        "pra": pra,
        # Distractor fields
        "cholesterol": cholesterol,
        "bmi": bmi,
        "albumin": albumin,
        "hemoglobin": hemoglobin,
        "t1_snapshot": t1_snapshot,
    }
    if hba1c_t5 is not None and not missing_hba1c:
        patient["hba1c"] = hba1c_t5

    return patient


def to_tau2_patient(p: dict, anomaly: bool = False) -> dict:
    """Convert prana_env patient dict to tau2 LabResult format."""
    pid = p["patient_id"]

    def lab_history(t1_val, t5_val, anomaly_entry=None) -> list:
        entries = []
        if t1_val is not None:
            entries.append({"value": t1_val, "recorded_at": T1_NOMINAL_DATE.isoformat()})
        if anomaly_entry:
            entries.append(anomaly_entry)
        if t5_val is not None:
            entries.append({"value": t5_val, "recorded_at": T5_MEAS_DATE.isoformat()})
        return entries

    snap = p.get("t1_snapshot", {})

    # Anomaly: inject a second T5 measurement with >25% delta, 6 days before filing
    gfr_anomaly = None
    if anomaly and p.get("gfr") is not None:
        anomaly_gfr = round(p["gfr"] * 0.55, 1)  # 45% drop — clearly anomalous
        gfr_anomaly = {"value": anomaly_gfr, "recorded_at": "2026-03-01"}

    tau2: dict = {
        "patient_id": pid,
        "name": p["name"],
        "age": p["age"],
        "blood_type": p.get("blood_type"),
        "pra": p.get("pra"),
        "gfr": lab_history(snap.get("gfr"), p.get("gfr"), gfr_anomaly),
        "creatinine": lab_history(snap.get("creatinine"), p.get("creatinine")),
        "hba1c": lab_history(snap.get("hba1c"), p.get("hba1c")),
    }
    return tau2


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    prana_env_root = Path(__file__).parent.parent
    tau2_root = prana_env_root.parent / "tau2-bench"

    # Load existing P001-P003 as anchors
    existing_prana = json.loads((prana_env_root / "data" / "patient_db.json").read_text())
    existing_patients = existing_prana["patients"]  # P001, P002, P003

    # Load existing tau2 db to preserve non-patient sections
    tau2_db_path = tau2_root / "data" / "tau2" / "domains" / "prana" / "db.json"
    existing_tau2 = json.loads(tau2_db_path.read_text())

    # Generate P004-P050
    new_prana_patients = {}
    new_tau2_patients = {}

    for idx in range(4, N_PATIENTS + 1):
        p = generate_patient(idx)
        pid = p["patient_id"]
        is_anomaly = (idx - 4) in ANOMALY_PATIENT_INDICES

        # Add distractor fields to existing P001-P003 if not present
        new_prana_patients[pid] = p
        new_tau2_patients[pid] = to_tau2_patient(p, anomaly=is_anomaly)

    # Add distractor fields to existing P001-P003
    distractor_defaults = {
        "P001": {"cholesterol": 187.3, "bmi": 24.1, "albumin": 3.2, "hemoglobin": 10.8},
        "P002": {"cholesterol": 214.6, "bmi": 27.3, "albumin": 2.8, "hemoglobin": 9.4},
        "P003": {"cholesterol": 168.9, "bmi": 22.7, "albumin": 3.6, "hemoglobin": 11.2},
    }
    for pid, extras in distractor_defaults.items():
        existing_patients[pid].update(extras)

    # ── Write prana_env patient_db.json ───────────────────────────────────────
    prana_out = {"patients": {**existing_patients, **new_prana_patients}}
    out_path = prana_env_root / "data" / "patient_db.json"
    out_path.write_text(json.dumps(prana_out, indent=2))
    print(f"Wrote {len(prana_out['patients'])} patients → {out_path}")

    # ── Write tau2 db.json ─────────────────────────────────────────────────────
    all_tau2_patients = {**existing_tau2["patient_db"], **new_tau2_patients}
    existing_tau2["patient_db"] = all_tau2_patients
    tau2_db_path.write_text(json.dumps(existing_tau2, indent=2))
    print(f"Wrote {len(all_tau2_patients)} patients → {tau2_db_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    stages = {}
    missing_hba1c = missing_creatinine = missing_bt = anomaly_count = 0
    for pid, p in prana_out["patients"].items():
        s = p.get("ckd_stage", "?")
        stages[s] = stages.get(s, 0) + 1
        if p.get("hba1c") is None:
            missing_hba1c += 1
        if p.get("creatinine") is None:
            missing_creatinine += 1
        if p.get("blood_type") is None:
            missing_bt += 1

    print(f"\nSummary:")
    print(f"  CKD stages: {dict(sorted(stages.items()))}")
    print(f"  Missing HbA1c:    {missing_hba1c}/{N_PATIENTS}")
    print(f"  Missing creatinine: {missing_creatinine}/{N_PATIENTS}")
    print(f"  Missing blood_type: {missing_bt}/{N_PATIENTS}")
    print(f"  Anomaly patients (tau2): {len(ANOMALY_PATIENT_INDICES)}")


if __name__ == "__main__":
    main()
