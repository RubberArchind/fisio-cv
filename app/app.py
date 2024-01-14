from flask import Flask
from routes import Routes

app = Flask(__name__)

route = Routes(app)
route.setup()

# Run Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)