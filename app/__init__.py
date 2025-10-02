from flask import Flask
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine
import sqlite3
from flask_migrate import Migrate
from app.config import Config
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS

class UTF8JSONProvider(DefaultJSONProvider):
    ensure_ascii = False

db = SQLAlchemy()
migrate = Migrate()
socketio = SocketIO(cors_allowed_origins="*", logger=True, engineio_logger=True)

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
    socketio.init_app(app) 
    enable_sqlite_fk(app)

    from app.api.patients import patients_bp
    app.register_blueprint(patients_bp, url_prefix="/api/patients")
    
    from app.api.visits import visits_bp
    app.register_blueprint(visits_bp, url_prefix="/api/visits")

    from app.api.monitor import monitor_bp
    app.register_blueprint(monitor_bp, url_prefix="/api/monitor")

    from app.api.records import records_bp
    app.register_blueprint(records_bp, url_prefix="/api/records")

    from app.api.predictions import predictions_bp
    app.register_blueprint(predictions_bp, url_prefix="/api/predictions") 

    from app.ws.records import register_records_ws
    register_records_ws(socketio)
    CORS(app)
    return app
