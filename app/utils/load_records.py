from app import db
import sqlalchemy as sa
from sqlalchemy.orm import selectinload
from app.models import Patient, Visit, Record
from app.services import visits, patients, records

def load_from_csv(path: str, type: str = "FHR", patient_id: str = '321') -> Visit:
    import csv
    
    with open(path) as f:
        try:
            patients.create_patient(patient_id=patient_id)
        except Exception as e:
            pass
        rs = list(csv.DictReader(f))

        visit = visits.create_visit(patient_id=patient_id, start_time=1)
        for r in rs:
            records.add_record(visit_id=visit.visit_id, timestamp=float(r["time_sec"]), value=float(r["value"]), record_type=type)
    