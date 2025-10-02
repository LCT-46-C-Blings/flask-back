python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ./app/motet
make build
cd ../../
flask db init
flask db migrate
flask db upgrade
python3 fetality.py