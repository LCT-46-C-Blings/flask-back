from typing import List
from flask import Blueprint, jsonify, request
import app.services.predictions as svc_predictions
from app.models import Patient
from app.utils.load_records import load_from_csv

predictions_bp = Blueprint("predictions", __name__)

@predictions_bp.get("")
@predictions_bp.get("/")
def prediction():
    visit_id = request.args.get("visit_id", None)

    try:
        pr = svc_predictions.get_prediction(visit_id=visit_id)
        return {"prediction": 
            {
                "id": pr.prediction_id,
                "result": pr.result
            }
        }
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409


