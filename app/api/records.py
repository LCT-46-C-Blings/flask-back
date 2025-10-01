
from flask import Blueprint, jsonify, request
import app.services.records as svc_records

records_bp = Blueprint("records", __name__)

@records_bp.get("/fhr")
def fhr():
    visit_id = request.args.get("visit_id", type=int)
    if not visit_id:
        return jsonify(ok=False, error="query param 'visit_id' is required"), 400

    try:
        rs = svc_records.list_records(visit_id=visit_id, record_type="FHR")
        return {"items": [{"timestamp": r.timestamp, "value": r.value} for r in rs]}
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409
    

@records_bp.get("/uc")
def uc():
    visit_id = request.args.get("visit_id", type=int)
    if not visit_id:
        return jsonify(ok=False, error="query param 'visit_id' is required"), 400

    try:
        rs = svc_records.list_records(visit_id=visit_id, record_type="UC")
        return {"items": [{"timestamp": r.timestamp, "value": r.value} for r in rs]}
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409