from typing import List, Dict
from app.model.inference import HypoxiaInference
import app.services.records as svc_records
import app.services.visits as svc_visits
import app.services.patients as svc_patients

def get_prediction(visit_id: int):
    uc = svc_records.list_records(visit_id=visit_id, record_type="UC")
    fhr = svc_records.list_records(visit_id=visit_id, record_type="FHR")

    uc_v = [r.value for r in uc]
    uc_t = [r.timestamp for r in uc]

    fhr_v = [r.value for r in fhr]
    fhr_t = [r.timestamp for r in fhr]

    v = svc_visits.get_visit(visit_id=visit_id)
    anamnesis = svc_patients.get_anamnesis(v.patient_id)
    print(anamnesis)
    inference = HypoxiaInference("app/model/hypoxia_model.pkl")
        
    result = inference.predict_from_arrays(
        fhr_timestamps=fhr_t,
        fhr_values=fhr_v,
        uc_timestamps=uc_t,
        uc_values=uc_v,
        medical_history="\n".join(anamnesis)
    )
    
    return result