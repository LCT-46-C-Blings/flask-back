from flask import Flask
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine
import sqlite3
from flask_migrate import Migrate
from app.config import Config


db = SQLAlchemy()
migrate = Migrate()
socketio = SocketIO(async_mode="eventlet", cors_allowed_origins="*")

def enable_sqlite_fk(app):
    @event.listens_for(Engine, "connect")
    def _set_sqlite_fk(dbapi_conn, _):
        if isinstance(dbapi_conn, sqlite3.Connection):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

def create_app(config_object=Config):
    app = Flask(__name__)
    app.config.from_object(config_object)

    db.init_app(app)
    migrate.init_app(app, db)

    from app.api.patients import patient_bp
    app.register_blueprint(patient_bp, url_prefix="/api/patients")

    from app.ws.records import register_records_ws
    register_records_ws(socketio)

    return app
