import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    FLASK_RUN_HOST = os.getenv("FLASK_RUN_HOST", "localhost")
    FLASK_RUN_PORT = os.getenv("FLASK_RUN_PORT", "5000")
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.getenv('DATABASE_URL', 'db.sqlite3')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
