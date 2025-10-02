from app import create_app

if __name__ == "__main__":
    a = create_app()
    a.run(host=a.config["FLASK_RUN_HOST"], port=a.config["FLASK_RUN_PORT"], debug=True)
    
    