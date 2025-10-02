from __future__ import annotations
import time
from typing import Optional, List
from sqlalchemy import select, delete as sa_delete
from sqlalchemy.exc import IntegrityError
from app import db
from app.models import Visit, Patient
from app.emulator.emulator import stop_emulator
from app.ws.records import _get_state, _safe_disconnect, _with_state

def create_visit(patient_id: int, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Visit:
    if not db.session.get(Patient, patient_id):
        raise ValueError("patient not found")
    if not start_time:
        start_time = time.time()
    if end_time and end_time < start_time:
        raise ValueError("end_time must be >= start_time")
    v = Visit(patient_id=patient_id, start_time=start_time, end_time=end_time)
    db.session.add(v)
    try:
        db.session.commit()
    except IntegrityError as e:
        db.session.rollback()
        raise ValueError("failed to create visit") from e
    return v

def finish_visit(visit_id: int, end_time: float) -> bool:
    v = db.session.get(Visit, visit_id)
    if not v:
        return False
    if end_time < v.start_time:
        raise ValueError("end_time must be >= start_time")
    v.end_time = end_time
    db.session.commit()
    
    status = _get_state("fhr", "visit_id")
    if status == visit_id:
        stop_emulator(graceful=True)
        _safe_disconnect("/ws/records/fhr", _get_state("fhr", "sid"))
        _safe_disconnect("/ws/records/uc",  _get_state("uc", "sid"))
        # обнуляем состояние, чтобы не возобновилось
        _with_state("fhr", "sid", None)
        _with_state("uc",  "sid", None)
        _with_state("fhr", "visit_id", None)
        _with_state("uc",  "visit_id", None)
        _with_state("fhr", "snapshot_sent", False)
        _with_state("uc",  "snapshot_sent", False)
    return True

def get_visit(visit_id: int) -> Optional[Visit]:
    return db.session.get(Visit, visit_id)

def list_visits(patient_id: Optional[int] = None, offset: int = 0, limit: int = None) -> List[Visit]:
    stmt = select(Visit)
    if patient_id:
        stmt = stmt.where(Visit.patient_id == patient_id)
    stmt = stmt.order_by(Visit.start_time).offset(offset).limit(limit)
    return db.session.scalars(stmt).all()

def delete_visit(visit_id: int) -> bool:
    v = db.session.get(Visit, visit_id)
    if not v:
        return False
    db.session.delete(v)
    db.session.commit()
    return True

def delete_all_visits() -> int:
    res = db.session.execute(sa_delete(Visit))
    db.session.commit()
    return int(res.rowcount or 0)
