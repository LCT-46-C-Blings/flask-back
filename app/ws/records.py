from __future__ import annotations
from flask import request
from flask_socketio import Namespace, emit
from sqlalchemy import event
from app import db, socketio
from app.models import Record
from app.services import visits as vis_svc, records as rec_svc
from app.emulator.emulator import (
    start_emulator, stop_emulator, set_on_finished, emulator_status
)

_state = {
    "fhr": {"sid": None, "visit_id": None},  # пульс (FHR)
    "uc":  {"sid": None, "visit_id": None},  # схватки (UC)
}

def _push(channel: str, event_name: str, payload: dict):
    sid = _state[channel]["sid"]
    if sid:
        socketio.emit(event_name, payload, to=sid, namespace=f"/ws/records/{channel}")

class _BaseNS(Namespace):
    channel = ""   # 'fhr' | 'uc'
    rtype   = ""   # 'FHR' | 'UC'

    def on_connect(self):
        st = _state[self.channel]
        st["sid"] = request.sid

        visit_id = request.args.get("visit_id", type=int)
        if not visit_id:
            emit("error", {"ok": False, "message": "visit_id required"})
            return False
        if not vis_svc.get_visit(visit_id):
            emit("error", {"ok": False, "message": "visit not found"})
            return False

        st["visit_id"] = visit_id

        # Эмулятор управляем ТОЛЬКО из FHR-канала
        if self.channel == "fhr":
            # Закрыть оба сокета, когда эмулятор завершится сам
            fhr_sid = request.sid
            def _on_emulator_finished(_payload: dict):
                try:
                    if _state["fhr"]["sid"]:
                        socketio.disconnect(sid=_state["fhr"]["sid"], namespace="/ws/records/fhr")
                except Exception:
                    pass
                try:
                    if _state["uc"]["sid"]:
                        socketio.disconnect(sid=_state["uc"]["sid"], namespace="/ws/records/uc")
                except Exception:
                    pass

            set_on_finished(_on_emulator_finished)

            # Запустить/перезапустить при необходимости
            status = emulator_status()
            need_restart = (not status.get("running")) or (status.get("visit_id") != visit_id)
            if need_restart:
                start_emulator(
                    visit_id=visit_id,
                    loop=(request.args.get("loop", "false").lower() == "true"),
                    force_restart=True,
                )

        emit("connected", {"ok": True, "visit_id": visit_id, "type": self.rtype})

        # Снапшот только своего типа
        items = rec_svc.list_records(visit_id, record_type=self.rtype)
        emit(f"{self.channel}:snapshot", {
            "items": [{"timestamp": r.timestamp, "value": r.value} for r in items],
        })

    def on_disconnect(self):
        st = _state[self.channel]
        if st["sid"] == request.sid:
            st["sid"] = None
            st["visit_id"] = None

        # Гасим эмулятор, когда уходит FHR-клиент (основной контролёр)
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

    # Пульс (FHR)
    s_fhr = _state["fhr"]
    if s_fhr["sid"] and s_fhr["visit_id"] is not None:
        for r in recs:
            if r.record_type == "FHR" and r.visit_id == s_fhr["visit_id"]:
                _push("fhr", "fhr:new", {
                    "timestamp": r.timestamp,
                    "value": r.value,
                })

    # Схватки (UC)
    s_uc = _state["uc"]
    if s_uc["sid"] and s_uc["visit_id"] is not None:
        for r in recs:
            if r.record_type == "UC" and r.visit_id == s_uc["visit_id"]:
                _push("uc", "uc:new", {
                    "timestamp": r.timestamp,
                    "value": r.value,
                })

def register_records_ws(socketio_instance):
    socketio_instance.on_namespace(RecordsFHRNS("/ws/records/fhr"))
    socketio_instance.on_namespace(RecordsUCNS("/ws/records/uc"))
