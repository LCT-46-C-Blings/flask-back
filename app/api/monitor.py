from __future__ import annotations
import time
from flask import Blueprint, request, jsonify
from app.services import records as rec_svc
from app.emulator.emulator import get_active_visit

monitor_bp = Blueprint("monitor", __name__)

@monitor_bp.post("/bpm")
def ingest_bpm():
    value = request.args.get("value", type=float)
    if value is None:
        return jsonify(ok=False, error="query param 'value' is required"), 400

    visit_id = get_active_visit()
    if not visit_id:
        return jsonify(ok=False, error="No active visit set for emulator"), 409

    ts = time.time()
    rec_svc.add_record(visit_id=visit_id, timestamp=ts, value=value, record_type="FHR")
    return jsonify(ok=True, visit_id=visit_id)

@monitor_bp.post("/uterus")
def ingest_uterus():
    value = request.args.get("value", type=float)
    if value is None:
        return jsonify(ok=False, error="query param 'value' is required"), 400

    visit_id = get_active_visit()
    if not visit_id:
        return jsonify(ok=False, error="No active visit set for emulator"), 409

    ts = time.time()
    rec_svc.add_record(visit_id=visit_id, timestamp=ts, value=value, record_type="UC")
    return jsonify(ok=True, visit_id=visit_id)
