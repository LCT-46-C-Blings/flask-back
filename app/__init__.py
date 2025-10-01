from flask import Flask
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine
import sqlite3
from flask_migrate import Migrate
from app.config import Config
from flask.json.provider import DefaultJSONProvider

class UTF8JSONProvider(DefaultJSONProvider):
    ensure_ascii = False

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
    app.json = UTF8JSONProvider(app)

    db.init_app(app)
    migrate.init_app(app, db)

    from app.api.patients import patients_bp
    app.register_blueprint(patients_bp, url_prefix="/api/patients")
    
    from app.api.visits import visits_bp
    app.register_blueprint(visits_bp, url_prefix="/api/visits")

    from app.ws.records import register_records_ws
    register_records_ws(socketio)

    from app.ws.events import register_ws
    register_ws(socketio)

    return app
