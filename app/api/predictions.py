from typing import List
from flask import Blueprint, jsonify, request
import app.services.predictions as svc_predictions
from app.models import Patient
from app.utils.load_records import load_from_csv
from app.model.predict import get_prediction

predictions_bp = Blueprint("predictions", __name__)

@predictions_bp.get("")
@predictions_bp.get("/")
def prediction():
    visit_id = request.args.get("visit_id", None)

    try:
        pr = svc_predictions.get_prediction_by_visit_id(visit_id=visit_id)
        if not pr:
            pr = get_prediction(visit_id=visit_id)
            return {"prediction": 
                {
                    "id": 0,
                    "result": pr
                }
            }
        return {"prediction": 
            {
                "id": pr.prediction_id,
                "result": pr.result
            }
        }
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409

@predictions_bp.get("/predict")
def predict():
    visit_id = request.args.get("visit_id")
    print(visit_id)
    try:
        res = get_prediction(visit_id=visit_id)
        return {"prediction": 
            {
                "result": res,
            }
        }
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409