from __future__ import annotations
from typing import List, Optional
import time
from sqlalchemy import select, delete as sa_delete, func, distinct
from app import db
from app.models import Prediction, Visit

def _fetch_visits(visit_ids: List[int]) -> list[Visit]:
    if not visit_ids:
        raise ValueError("visit_ids is empty")
    visits = db.session.scalars(select(Visit).where(Visit.visit_id.in_(visit_ids))).all()
    if len(visits) != len(set(visit_ids)):
        found = {v.visit_id for v in visits}
        missing = [vid for vid in set(visit_ids) if vid not in found]
        raise ValueError(f"visit(s) not found: {', '.join(missing)}")
    return visits

def _require_same_patient(visits: list[Visit]) -> None:
    pids = {v.patient_id for v in visits}
    if len(pids) != 1:
        raise ValueError("all visit_ids must belong to the same patient")

def _apply_filters(
    patient_id: Optional[int],
    visit_ids: Optional[List[int]],
    visits_mode: str,  # "at_least" | "only"
):
    stmt = select(Prediction)

    if patient_id or visit_ids:
        stmt = stmt.join(Prediction.visits)

    if patient_id:
        stmt = stmt.where(Visit.patient_id == patient_id)

    if visit_ids:
        ids = set(visit_ids)
        n = len(ids)
        base = (
            stmt.where(Visit.visit_id.in_(ids))
                .group_by(Prediction.prediction_id)
                .having(func.count(distinct(Visit.visit_id)) == n)
        )

        if visits_mode == "at_least":
            stmt = base
        elif visits_mode == "only":
            totals = (
                select(
                    Prediction.prediction_id.label("pid"),
                    func.count(distinct(Visit.visit_id)).label("total_cnt"),
                )
                .select_from(Prediction)
                .join(Prediction.visits)
                .group_by(Prediction.prediction_id)
            ).subquery()
            stmt = base.join(totals, totals.c.pid == Prediction.prediction_id)\
                       .having(totals.c.total_cnt == n)
        else:
            raise ValueError("visits_mode must be 'at_least' or 'only'")

    return stmt


def create_prediction(visit_ids: List[int], result: str, created_at: Optional[float] = None) -> Prediction:
    visits = _fetch_visits(visit_ids)
    _require_same_patient(visits)
    p = Prediction(       
        result=result,
        created_at=(created_at or time.time()),
        visits=visits,
    )
    db.session.add(p)
    db.session.commit()
    return p

def get_prediction(prediction_id: int) -> Optional[Prediction]:
    return db.session.get(Prediction, prediction_id)

def update_prediction_result(prediction_id: int, result: str) -> bool:
    p = db.session.get(Prediction, prediction_id)
    if not p:
        return False
    p.result = result
    db.session.commit()
    return True

def set_prediction_visits(prediction_id: int, visit_ids: List[str]) -> bool:
    p = db.session.get(Prediction, prediction_id)
    if not p:
        return False
    visits = _fetch_visits(visit_ids)
    _require_same_patient(visits)
    p.visits = visits
    db.session.commit()
    return True

def list_predictions(
    patient_id: Optional[str] = None,
    visit_ids: Optional[List[str]] = None,
    visits_mode: str = "at_least",        # "at_least" | "only"
) -> List[Prediction]:
    stmt = _apply_filters(patient_id, visit_ids, visits_mode)
    return db.session.scalars(stmt).all()

def list_predictions_by_visits(visit_ids: List[int], visits_mode: str = "at_least") -> List[Prediction]:
    return list_predictions(patient_id=None,visit_ids=visit_ids,visits_mode=visits_mode)

def list_predictions_by_patient(patient_id: int) -> List[Prediction]:
    return list_predictions(patient_id=patient_id,visit_ids=None)


def delete_prediction(prediction_id: int) -> bool:
    p = db.session.get(Prediction, prediction_id)
    if not p:
        return False
    db.session.delete(p)
    db.session.commit()
    return True

def delete_predictions(patient_id: Optional[int] = None,
                       visit_ids: Optional[List[int]] = None, visits_mode: str = "at_least") -> int:
    stmt = _apply_filters(patient_id, visit_ids, visits_mode)
    ids_stmt = stmt.with_only_columns(Prediction.prediction_id)\
                   .group_by(Prediction.prediction_id)
    ids = [pid for (pid,) in db.session.execute(ids_stmt).all()]
    if not ids:
        return 0
    res = db.session.execute(
        sa_delete(Prediction).where(Prediction.prediction_id.in_(ids))
    )
    db.session.commit()
    return int(res.rowcount or 0)

def delete_predictions_by_visits(visit_ids: List[int], visits_mode: str = "at_least") -> int:
    return delete_predictions(patient_id=None, visit_ids=visit_ids, visits_mode=visits_mode)

def delete_predictions_by_patient(patient_id: int) -> int:
    return delete_predictions(patient_id=patient_id, visit_ids=None)
