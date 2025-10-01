from typing import List
from flask import Blueprint, jsonify, request
import app.services.patients as svc_patients
import app.services.visits as svc_visits
from app.models import Patient
from app.utils.load_records import load_from_csv

visits_bp = Blueprint("visits", __name__)

@visits_bp.post("/create")
def create_visit():
    patient_id = request.args.get("patient_id", None)
    
    try:
        v = svc_visits.create_visit(patient_id=patient_id)
        return jsonify({
            "ok": True,
            "visit_id": v.visit_id,
            "created": True,
        }), 201
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409


@visits_bp.get("/list")
def index():
    patient_id = request.args.get("patient_id", None)

    try:
        return {"visits": [
            {
                "id": a.visit_id,
                "start_time": a.start_time,
                "end_time": a.end_time
            }
            for a in svc_visits.list_visits(patient_id)
        ]}
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409


