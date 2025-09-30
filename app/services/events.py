from __future__ import annotations
from typing import Optional, List
from sqlalchemy import select, delete as sa_delete
from app import db
from app.models import Event, Visit

def add_event(visit_id: str, event_type: str, start: float,
              end: Optional[float] = None, value: Optional[float] = None,
              event_id: Optional[str] = None) -> Event:
    if not db.session.get(Visit, visit_id):
        raise ValueError("visit not found")

    e = Event(event_id=event_id, visit_id=visit_id, event_type=event_type, start=start, end=end, value=value)
    db.session.add(e)
    db.session.commit()
    return e

def list_events(visit_id: str) -> List[Event]:
    stmt = select(Event).where(Event.visit_id == visit_id).order_by(Event.start)
    return db.session.scalars(stmt).all()

def delete_event(event_id: str) -> bool:
    e = db.session.get(Event, event_id)
    if not e:
        return False
    db.session.delete(e)
    db.session.commit()
    return True

def delete_events_by_visit(visit_id: str) -> int:
    res = db.session.execute(sa_delete(Event).where(Event.visit_id == visit_id))
    db.session.commit()
    return int(res.rowcount or 0)
