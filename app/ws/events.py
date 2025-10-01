from __future__ import annotations
from flask import request
from flask_socketio import Namespace, emit
from sqlalchemy import event as sa_event
from app import db, socketio
from app.models import Event 
from app.services import visits as vis_svc, events as ev_svc

_current_sid: str | None = None
_current_visit_id: str | None = None

def _push(event: str, payload: dict):
    if _current_sid:
        socketio.emit(event, payload, to=_current_sid, namespace="/ws/events")

class EventsNS(Namespace):
    def on_connect(self):
        global _current_sid, _current_visit_id
        _current_sid = self.sid

        visit_id = request.args.get("visit_id", type=str)
        if not visit_id:
            emit("error", {"ok": False, "message": "visit_id required"})
            return
        if not vis_svc.get_visit(visit_id):
            emit("error", {"ok": False, "message": "visit not found"})
            return

        _current_visit_id = visit_id
        emit("connected", {"ok": True, "visit_id": visit_id})

        items = ev_svc.list_events(visit_id)
        emit("events:snapshot", {
            "visit_id": visit_id,
            "items": [{"start": e.start, "end": e.end, "event_type": e.event_type} for e in items],
        })

    def on_disconnect(self):
        global _current_sid, _current_visit_id
        if _current_sid == self.sid:
            _current_sid = None
            _current_visit_id = None

@sa_event.listens_for(db.session, "after_flush")
def _collect_new_events(session, _):
    new_events = [o for o in session.new if isinstance(o, Event)]
    if new_events:
        session.info.setdefault("ws_new_events", []).extend(new_events)

@sa_event.listens_for(db.session, "after_commit")
def _broadcast_new_events(session):
    events = session.info.pop("ws_new_events", [])
    if not events or not _current_sid or not _current_visit_id:
        return
    for e in events:
        if e.visit_id == _current_visit_id:
            _push("event:new", {"item": {"start": e.start, "end": e.end, "event_type": e.event_type}})

def register_ws(socketio_instance):
    socketio_instance.on_namespace(EventsNS("/ws/events"))
