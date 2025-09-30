from __future__ import annotations
from typing import Optional, List

import sqlalchemy as sa
import sqlalchemy.orm as so
from sqlalchemy import CheckConstraint
from app import db
from app.services.id import new_id

prediction_visit = db.Table(
    "prediction_visit",
    db.Column("prediction_id", 
              db.String, 
              db.ForeignKey("prediction.prediction_id", ondelete="CASCADE"), 
              primary_key=True),
    db.Column("visit_id",      
              db.String, 
              db.ForeignKey("visit.visit_id", ondelete="CASCADE"), 
              primary_key=True),
)

class PatientAnamnesis(db.Model):
    __tablename__ = "patient_anamnesis"

    anamnesis_id: so.Mapped[int] = so.mapped_column(db.String, primary_key=True, default=new_id)
    patient_id: so.Mapped[str] = so.mapped_column(
        db.String,
        db.ForeignKey("patient.patient_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    text: so.Mapped[str] = so.mapped_column(db.String, nullable=False)

    patient: so.Mapped["Patient"] = so.relationship("Patient", back_populates="anamnesis_lines")


class Patient(db.Model):
    __tablename__ = "patient"

    patient_id: so.Mapped[str] = so.mapped_column(db.String, primary_key=True, default=new_id)

    visits: so.Mapped[List["Visit"]] = so.relationship(
        "Visit",
        back_populates="patient",
    )

    anamnesis_lines: so.Mapped[List["PatientAnamnesis"]] = so.relationship(
        "PatientAnamnesis",
        back_populates="patient",
    )

class Visit(db.Model):
    __tablename__ = "visit"

    visit_id: so.Mapped[int] = so.mapped_column(db.String, primary_key=True, default=new_id)

    patient_id: so.Mapped[str] = so.mapped_column(
        db.String,
        db.ForeignKey("patient.patient_id", ondelete="CASCADE"),
        nullable=False,
    )

    start_time: so.Mapped[float] = so.mapped_column(db.Float, nullable=False)
    end_time:   so.Mapped[Optional[float]] = so.mapped_column(db.Float)

    patient: so.Mapped["Patient"] = so.relationship(
        "Patient",
        back_populates="visits",
    )

    records: so.Mapped[List["Record"]] = so.relationship(
        "Record",
        back_populates="visit",
    )

    events: so.Mapped[List["Event"]] = so.relationship(
        "Event",
        back_populates="visit",
    )

    predictions: so.Mapped[List["Prediction"]] = so.relationship(
        "Prediction",
        secondary=lambda: prediction_visit,
        back_populates="visits",
    )

class Record(db.Model):
    __tablename__ = "record"

    record_id:  so.Mapped[str]   = so.mapped_column(db.String, primary_key=True, default=new_id)
    timestamp:  so.Mapped[float] = so.mapped_column(db.Float, nullable=False)
    value:      so.Mapped[float] = so.mapped_column(db.Float, nullable=False)
    record_type: so.Mapped[str]  = so.mapped_column(db.String, nullable=False)  # 'FHR'|'UC'

    visit_id: so.Mapped[int] = so.mapped_column(
        db.String,
        db.ForeignKey("visit.visit_id", ondelete="CASCADE"),
        nullable=False,
    )

    visit: so.Mapped["Visit"] = so.relationship(
        "Visit",
        back_populates="records",
    )

    # __table_args__ = (
    #     CheckConstraint("record_type IN ('FHR','UC')", name="ck_record_type"),
    # )

class Event(db.Model):
    __tablename__ = "event"

    event_id:  so.Mapped[str]   = so.mapped_column(db.String, primary_key=True, default=new_id)

    visit_id:  so.Mapped[str]   = so.mapped_column(
        db.String,
        db.ForeignKey("visit.visit_id", ondelete="CASCADE"),
        nullable=False,
    )

    event_type: so.Mapped[str]   = so.mapped_column(db.String, nullable=False)
    start:      so.Mapped[float] = so.mapped_column(db.Float, nullable=False)
    end:        so.Mapped[Optional[float]] = so.mapped_column(db.Float)
    value:      so.Mapped[Optional[float]] = so.mapped_column(db.Float)

    visit: so.Mapped["Visit"] = so.relationship(
        "Visit",
        back_populates="events",
    )

class Prediction(db.Model):
    __tablename__ = "prediction"

    prediction_id: so.Mapped[str]   = so.mapped_column(db.String, primary_key=True, default=new_id)
    created_at:    so.Mapped[float] = so.mapped_column(db.Float, nullable=False)
    result:        so.Mapped[str]   = so.mapped_column(db.String, nullable=False)

    visits: so.Mapped[List["Visit"]] = so.relationship(
        "Visit",
        secondary=prediction_visit,
        back_populates="predictions",
    )
