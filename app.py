from api import create_app
from api.config import FLASK_DEBUG, FLASK_PORT

app = create_app()

if __name__ == '__main__':
    app.run(debug=FLASK_DEBUG, host='0.0.0.0', port=FLASK_PORT)
