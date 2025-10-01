from __future__ import annotations
from typing import Optional, List
from flask import Blueprint
from sqlalchemy import select, delete as sa_delete
from sqlalchemy.exc import IntegrityError
from app import db
from app.models import Patient, PatientAnamnesis

patients_bp    = Blueprint("patients", __name__)

def create_patient() -> Patient:
    p = Patient()

    db.session.add(p)
    try:
        db.session.commit()
    except IntegrityError as e:
        db.session.rollback()
        raise ValueError("failed to create patient") from e
    return p

def get_patient(patient_id: int) -> Optional[Patient]:
    return db.session.get(Patient, patient_id)

def list_patients(offset: int = 0, limit: int = None) -> List[Patient]:
    stmt = select(Patient).order_by(Patient.patient_id).offset(offset).limit(limit)
    return db.session.scalars(stmt).all()

def delete_patient(patient_id: int) -> bool:
    obj = db.session.get(Patient, patient_id)
    if not obj:
        return False
    db.session.delete(obj)
    db.session.commit()
    return True

def delete_all_patients() -> int:
    res = db.session.execute(sa_delete(Patient))
    db.session.commit()
    return int(res.rowcount or 0)

def get_anamnesis(patient_id: int) -> list[str]:
    if not db.session.get(Patient, patient_id):
        return []
    stmt = (select(PatientAnamnesis)
            .where(PatientAnamnesis.patient_id == patient_id)
            .order_by(PatientAnamnesis.anamnesis_id))
    return [r.text for r in db.session.scalars(stmt).all()]

def set_anamnesis(patient_id: int, lines: list[str]) -> None:
    if not db.session.get(Patient, patient_id):
        raise ValueError("patient not found")
    db.session.execute(
        sa_delete(PatientAnamnesis).where(PatientAnamnesis.patient_id == patient_id)
    )
    db.session.add_all([PatientAnamnesis(patient_id=patient_id, text=t) for t in lines])
    db.session.commit()

def append_anamnesis_line(patient_id: int, text: str) -> None:
    if not db.session.get(Patient, patient_id):
        raise ValueError("patient not found")
    db.session.add(PatientAnamnesis(patient_id=patient_id, text=text))
    db.session.commit()
