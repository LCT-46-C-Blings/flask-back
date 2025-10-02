from __future__ import annotations

import threading
from typing import Dict, Any, Optional

from flask import request
from flask_socketio import Namespace, emit
from sqlalchemy import event

from app import db, socketio
from app.models import Record
from app.services import visits as vis_svc, records as rec_svc
from app.emulator.emulator import (
    start_emulator,
    stop_emulator,
    set_on_finished,
    emulator_status,
)

_state_lock = threading.Lock()
_state: Dict[str, Dict[str, Optional[str] | Optional[int] | bool]] = {
    "fhr": {"sid": None, "visit_id": None, "snapshot_sent": False},
    "uc":  {"sid": None, "visit_id": None, "snapshot_sent": False},
}

def _with_state(channel: str, key: str, value):
    with _state_lock:
        _state[channel][key] = value

def _get_state(channel: str, key: str):
    with _state_lock:
        return _state[channel][key]

def _push(channel: str, event_name: str, payload: dict):
    sid = _get_state(channel, "sid")
    if not sid:
        return
    socketio.emit(event_name, payload, to=sid, namespace=f"/ws/records/{channel}")

def _safe_disconnect(ns: str, sid: Optional[str]):
    if not sid:
        return
    socketio.emit("emulator:finished", {"message": "Эмулятор завершился"}, to=sid, namespace=ns)
    socketio.server.disconnect(sid, namespace=ns)

class _BaseNS(Namespace):
    channel = ""
    rtype   = ""

    def _visit_id_from_request(self) -> Optional[int]:
        vid = request.args.get("visit_id", type=int)
        if vid:
            return vid
        x_vid = request.headers.get("X-Visit-Id")
        if x_vid and x_vid.isdigit():
            return int(x_vid)
        return None

    def on_connect(self):
        visit_id = self._visit_id_from_request()
        if not visit_id:
            emit("error", {"ok": False, "message": "visit_id required"})
            return False
        if not vis_svc.get_visit(visit_id):
            emit("error", {"ok": False, "message": "visit not found"})
            return False

        old_visit = _get_state(self.channel, "visit_id")
        _with_state(self.channel, "sid", request.sid)
        _with_state(self.channel, "visit_id", int(visit_id))
        if old_visit != visit_id:
            _with_state(self.channel, "snapshot_sent", False)

        if self.channel == "fhr":
            def _on_emulator_finished(_payload: Dict[str, Any]):
                fhr_sid = _get_state("fhr", "sid")
                uc_sid  = _get_state("uc", "sid")
                _safe_disconnect("/ws/records/fhr", fhr_sid)
                _safe_disconnect("/ws/records/uc",  uc_sid)
            set_on_finished(_on_emulator_finished)
            status = emulator_status()
            need_restart = (not status.get("running")) or (status.get("visit_id") != visit_id)
            if need_restart:
                loop_flag = (request.args.get("loop", "false").lower() == "true")
                start_emulator(visit_id=visit_id, loop=loop_flag, force_restart=True)
                emit("info", {"message": "Эмулятор запущен"})

        emit("connected", {"ok": True, "visit_id": visit_id, "type": self.rtype})

        want_resend = request.args.get("resend_snapshot", "false").lower() == "true"
        already_sent = bool(_get_state(self.channel, "snapshot_sent"))
        if not already_sent or want_resend:
            items = rec_svc.list_records(visit_id, record_type=self.rtype)
            emit(f"{self.channel}:snapshot", {
                "items": [{"timestamp": r.timestamp, "value": r.value} for r in items],
            })
            if not want_resend:
                _with_state(self.channel, "snapshot_sent", True)

    def on_disconnect(self):
        sid_now = request.sid
        sid_saved = _get_state(self.channel, "sid")
        if sid_saved == sid_now:
            _with_state(self.channel, "sid", None)
            _with_state(self.channel, "visit_id", None)
            _with_state(self.channel, "snapshot_sent", False)
        if self.channel == "fhr":
            stop_emulator(graceful=True)

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
    s_fhr_sid = _get_state("fhr", "sid")
    s_fhr_vid = _get_state("fhr", "visit_id")
    if s_fhr_sid and s_fhr_vid is not None:
        for r in recs:
            if r.record_type == "FHR" and r.visit_id == s_fhr_vid:
                _push("fhr", "fhr:new", {"timestamp": r.timestamp, "value": r.value})
    s_uc_sid = _get_state("uc", "sid")
    s_uc_vid = _get_state("uc", "visit_id")
    if s_uc_sid and s_uc_vid is not None:
        for r in recs:
            if r.record_type == "UC" and r.visit_id == s_uc_vid:
                _push("uc", "uc:new", {"timestamp": r.timestamp, "value": r.value})

def register_records_ws(socketio_instance=None) -> None:
    sio = socketio_instance or socketio
    sio.on_namespace(RecordsFHRNS("/ws/records/fhr"))
    sio.on_namespace(RecordsUCNS("/ws/records/uc"))
