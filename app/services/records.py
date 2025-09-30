from __future__ import annotations
from typing import List, Optional
from sqlalchemy import select, delete as sa_delete
from app import db
from app.models import Record, Visit

_ALLOWED = {"FHR", "UC"}

def add_record(visit_id: str, timestamp: float, value: float, record_type: str,
               record_id: Optional[str] = None) -> Record:
    if not db.session.get(Visit, visit_id):
        raise ValueError("visit not found")
    if record_type not in _ALLOWED:
        raise ValueError("record_type must be 'FHR' or 'UC'")

    r = Record(record_id=record_id, visit_id=visit_id, timestamp=timestamp, value=value, record_type=record_type)
    db.session.add(r)
    db.session.commit()
    return r

def list_records(visit_id: str, record_type: Optional[str] = None,
                 ts_from: Optional[float] = None, ts_to: Optional[float] = None) -> List[Record]:
    stmt = select(Record).where(Record.visit_id == visit_id)
    if record_type:
        stmt = stmt.where(Record.record_type == record_type)
    if ts_from is not None:
        stmt = stmt.where(Record.timestamp >= ts_from)
    if ts_to is not None:
        stmt = stmt.where(Record.timestamp <= ts_to)
    stmt = stmt.order_by(Record.timestamp)
    return db.session.scalars(stmt).all()

def delete_record(record_id: str) -> bool:
    r = db.session.get(Record, record_id)
    if not r:
        return False
    db.session.delete(r)
    db.session.commit()
    return True

def delete_records_by_visit(visit_id: str) -> int:
    res = db.session.execute(sa_delete(Record).where(Record.visit_id == visit_id))
    db.session.commit()
    return int(res.rowcount or 0)
