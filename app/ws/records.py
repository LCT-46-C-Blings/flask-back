from __future__ import annotations

import threading
import logging
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
_state: Dict[str, Dict[str, Optional[int]]] = {
    "fhr": {"sid": None, "visit_id": None},  # пульс (FHR)
    "uc":  {"sid": None, "visit_id": None},  # схватки (UC)
}

log = logging.getLogger("ws.records")


def _with_state(channel: str, key: str, value: Optional[int]) -> None:
    with _state_lock:
        _state[channel][key] = value


def _get_state(channel: str, key: str) -> Optional[int]:
    with _state_lock:
        return _state[channel][key] 


def _push(channel: str, event_name: str, payload: dict):
    """Безопасная эмиссия в конкретного клиента канала."""
    sid = _get_state(channel, "sid")
    if not sid:
        return
    try:
        socketio.emit(event_name, payload, to=sid, namespace=f"/ws/records/{channel}")
    except Exception as e:
        log.exception("emit failed: channel=%s event=%s err=%s", channel, event_name, e)


def _get_visit_id_from_request() -> Optional[int]:
    """
    Достаём visit_id из query string. Клиент должен подключаться как:
    io('/ws/records/fhr?visit_id=1&loop=true', ...)
    """
    vid = request.args.get("visit_id", type=int)
    if vid:
        return vid
    x_vid = request.headers.get("X-Visit-Id")
    if x_vid and x_vid.isdigit():
        return int(x_vid)
    return None


class _BaseNS(Namespace):
    channel = ""   # 'fhr' | 'uc'
    rtype   = ""   # 'FHR' | 'UC'

    def on_connect(self):
        visit_id = _get_visit_id_from_request()
        if not visit_id:
            emit("error", {"ok": False, "message": "visit_id required"})
            log.warning("[%s] connect rejected: no visit_id", self.channel)
            return False

        if not vis_svc.get_visit(visit_id):
            emit("error", {"ok": False, "message": "visit not found"})
            log.warning("[%s] connect rejected: visit %s not found", self.channel, visit_id)
            return False

        _with_state(self.channel, "sid", request.sid)     
        _with_state(self.channel, "visit_id", int(visit_id))
        log.info("[%s] connected sid=%s visit_id=%s", self.channel, request.sid, visit_id)

        if self.channel == "fhr":
            def _on_emulator_finished(_payload: Dict[str, Any]):
                try:
                    fhr_sid = _get_state("fhr", "sid")
                    if fhr_sid:
                        socketio.disconnect(sid=fhr_sid, namespace="/ws/records/fhr")
                except Exception:
                    pass
                try:
                    uc_sid = _get_state("uc", "sid")
                    if uc_sid:
                        socketio.disconnect(sid=uc_sid, namespace="/ws/records/uc")
                except Exception:
                    pass
                log.info("[emulator] finished; both namespaces disconnected")

            set_on_finished(_on_emulator_finished)

            status = emulator_status()
            need_restart = (not status.get("running")) or (status.get("visit_id") != visit_id)
            if need_restart:
                try:
                    start_emulator(
                        visit_id=visit_id,
                        loop=False,
                        force_restart=True,
                    )
                    log.info("[emulator] started visit_id=%s loop=%s", visit_id, False)
                except FileNotFoundError as e:
                    emit("error", {"ok": False, "message": f"emulator files not found: {e}"})
                    log.exception("[emulator] start failed: %s", e)
                except Exception as e:
                    emit("error", {"ok": False, "message": f"emulator start error: {e}"})
                    log.exception("[emulator] start failed: %s", e)

            # def auto_stop():
            #     try:
            #         stop_emulator(graceful=True)
            #         log.info("[emulator] auto-stopped after 10s")
            #     except Exception as e:
            #         log.exception("auto-stop failed: %s", e)

            # threading.Timer(10.0, auto_stop).start()

        emit("connected", {"ok": True, "visit_id": visit_id, "type": self.rtype})

        try:
            items = rec_svc.list_records(visit_id, record_type=self.rtype)
            emit(f"{self.channel}:snapshot", {
                "items": [{"timestamp": r.timestamp, "value": r.value} for r in items],
            })
            log.debug("[%s] snapshot sent: %d items", self.channel, len(items))
        except Exception as e:
            emit("error", {"ok": False, "message": f"snapshot error: {e}"})
            log.exception("[%s] snapshot failed: %s", self.channel, e)

        def _safe_disconnect(ns: str, sid: Optional[str]):
            if not sid:
                return
            try:
                socketio.emit("emulator:finished", {"message": "Эмулятор завершился"}, to=sid, namespace=ns)
            except Exception:
                pass
            try:
                socketio.server.disconnect(sid, namespace=ns)
            except Exception:
                pass

        def _on_emulator_finished(_payload: dict):
            try:
                fhr_sid = _get_state("fhr", "sid")
                uc_sid  = _get_state("uc", "sid")
                _safe_disconnect("/ws/records/fhr", fhr_sid)
                _safe_disconnect("/ws/records/uc",  uc_sid)
            except Exception:
                log.exception("failed to disconnect on emulator finish")

        set_on_finished(_on_emulator_finished)


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
    s_fhr_sid = _get_state("fhr", "sid")
    s_fhr_vid = _get_state("fhr", "visit_id")
    if s_fhr_sid and s_fhr_vid is not None:
        for r in recs:
            if r.record_type == "FHR" and r.visit_id == s_fhr_vid:
                _push("fhr", "fhr:new", {"timestamp": r.timestamp, "value": r.value})

    # Схватки (UC)
    s_uc_sid = _get_state("uc", "sid")
    s_uc_vid = _get_state("uc", "visit_id")
    if s_uc_sid and s_uc_vid is not None:
        for r in recs:
            if r.record_type == "UC" and r.visit_id == s_uc_vid:
                _push("uc", "uc:new", {"timestamp": r.timestamp, "value": r.value})


def register_records_ws(socketio_instance):
    """Вызывай из create_app(): register_records_ws(socketio)"""
    socketio_instance.on_namespace(RecordsFHRNS("/ws/records/fhr"))
    socketio_instance.on_namespace(RecordsUCNS("/ws/records/uc"))
    log.info("namespaces registered: /ws/records/fhr, /ws/records/uc")
