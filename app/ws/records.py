from __future__ import annotations
from flask import request
from flask_socketio import Namespace, emit
from sqlalchemy import event
from app import db, socketio
from app.models import Record
from app.services import visits as vis_svc, records as rec_svc

_state = {
    "fhr": {"sid": None, "visit_id": None},  # пульс (FHR)
    "uc":  {"sid": None, "visit_id": None},  # схватки (UC)
}

def _push(channel: str, event_name: str, payload: dict):
    sid = _state[channel]["sid"]
    if sid:
        socketio.emit(event_name, payload, to=sid, namespace=f"/ws/{channel}")

class _BaseNS(Namespace):
    channel = ""   # 'fhr' | 'uc'
    rtype   = ""   # 'FHR' | 'UC'

    def on_connect(self):
        st = _state[self.channel]
        st["sid"] = self.sid

        visit_id = request.args.get("visit_id", type=str)
        if not visit_id:
            emit("error", {"ok": False, "message": "visit_id required"})
            return
        if not vis_svc.get_visit(visit_id):
            emit("error", {"ok": False, "message": "visit not found"})
            return

        st["visit_id"] = visit_id
        emit("connected", {"ok": True, "visit_id": visit_id, "type": self.rtype})

        # снапшот только своего типа
        items = rec_svc.list_records(visit_id, record_type=self.rtype)
        emit("{elf.rtype}:snapshot", {
            "items": [{"timestamp": r.timestamp, "value": r.value} for r in items],
        })

    def on_disconnect(self):
        st = _state[self.channel]
        if st["sid"] == self.sid:
            st["sid"] = None
            st["visit_id"] = None

class RecordsFHRNS(_BaseNS):
    channel = "fhr"
    rtype   = "FHR"

class RecordsUCNS(_BaseNS):
    channel = "uc"
    rtype   = "UC"

@event.listens_for(db.session, "after_flush")
def _collect_new_records(session, _):
    recs = [o for o in session.new if isinstance(o, Record)]
    if recs:
        session.info.setdefault("ws_new_records", []).extend(recs)

@event.listens_for(db.session, "after_commit")
def _broadcast_new_records(session):
    recs = session.info.pop("ws_new_records", [])
    if not recs:
        return

    # Пульс (FHR)
    s_fhr = _state["fhr"]
    if s_fhr["sid"] and s_fhr["visit_id"]:
        for r in recs:
            if r.record_type == "FHR" and r.visit_id == s_fhr["visit_id"]:
                _push("fhr", "fhr:new", {
                    "timestamp": r.timestamp,
                    "value": r.value,
                })

    # Схватки (UC)
    s_uc = _state["uc"]
    if s_uc["sid"] and s_uc["visit_id"]:
        for r in recs:
            if r.record_type == "UC" and r.visit_id == s_uc["visit_id"]:
                _push("uc", "uc:new", {
                    "timestamp": r.timestamp,
                    "value": r.value,
                })

def register_records_ws(socketio_instance):
    socketio_instance.on_namespace(RecordsFHRNS("/ws/records/fhr"))
    socketio_instance.on_namespace(RecordsUCNS("/ws/records/uc"))
