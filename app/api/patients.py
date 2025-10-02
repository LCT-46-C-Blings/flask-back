from typing import List
from flask import Blueprint, jsonify, request
from app.emulator.emulator import run_emulator, start_emulator
import app.services.patients as svc_patients
import app.services.visits as svc_visits
from app.models import Patient
from app.utils.load_records import load_from_csv

patients_bp = Blueprint("patients", __name__)

@patients_bp.post("/create")
def create_patient():
    data = request.get_json(silent=True)
    
    try:
        anamnesis = data.get("anamnesis", [])
        p = svc_patients.create_patient()
        svc_patients.set_anamnesis(patient_id=p.patient_id, lines=anamnesis)
        return jsonify({
            "ok": True,
            "patient_id": p.patient_id,
            "created": True,
        }), 201
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409

@patients_bp.post("set_anamnesis")
def set_anamnesis():
    data = request.get_json(silent=True)
    
    try:
        patient_id = request.args.get("patient_id", None)
        anamnesis = data.get("anamnesis", [])
        svc_patients.set_anamnesis(patient_id=patient_id, lines=anamnesis)
        return jsonify({
            "ok": True,
            "patient_id": patient_id,
            "created": True,
        }), 201
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409


@patients_bp.get("/list")
def list_patients():
    try:
        patients = svc_patients.list_patients()
        return {"patients": [
            {
                "id": p.patient_id,
                "anamnesis": svc_patients.get_anamnesis(p.patient_id),
                "appointments": [
                    {
                        "id": a.visit_id,
                        "start_time": a.start_time,
                        "end_time": a.end_time
                    }
                    for a in svc_visits.list_visits(p.patient_id)
                ],
            } 
            for p in patients
        ]}
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409
    
@patients_bp.get("/")
@patients_bp.get("")
def get_patient():
    patient_id = request.args.get("patient_id", None)
    try:
        p = svc_patients.get_patient(patient_id=patient_id)
        return {"patient": 
            {
                "id": p.patient_id,
                "anamnesis": svc_patients.get_anamnesis(p.patient_id),
                "appointments": [
                    {
                        "id": a.visit_id,
                        "start_time": a.start_time,
                        "end_time": a.end_time
                    }
                    for a in svc_visits.list_visits(p.patient_id)
                ],
            } 
        }
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409


@patients_bp.get("/test")
def test():
    svc_visits.create_visit(patient_id=1)

    # 3) Безопасный одноразовый прогон (гарантированная остановка)
    from time import sleep
    with run_emulator(visit_id=1, loop=False):
        sleep(3)