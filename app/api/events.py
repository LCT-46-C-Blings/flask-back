from flask import Blueprint, jsonify, request
import app.services.events as svc_events   

events_bp = Blueprint("events", __name__)

@events_bp.get("/")
@events_bp.get("")
def events():
    visit_id = request.args.get("visit_id", type=int)
    if not visit_id:
        return jsonify(ok=False, error="query param 'visit_id' is required"), 400

    try:
        es = svc_events.list_events(visit_id=visit_id)
        return {
            "events": 
                [
                    {
                        "id": e.event_id,
                        "type": e.event_type,
                        "start": e.start,
                        "end": e.end,
                        "value": e.value
                    }
                    for e in es
                ]
            }
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 409
