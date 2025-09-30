import os

class Config:
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.getenv('DATABASE_URL', 'db.sqlite3')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
