from app import create_app, socketio

if __name__ == "__main__":
    a = create_app()
    socketio.run(
        a,
        host=a.config["FLASK_RUN_HOST"],
        port=a.config["FLASK_RUN_PORT"],
        debug=True,
    )