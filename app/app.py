from flask import Flask
from routes import Routes

app = Flask(__name__)

route = Routes(app)
route.setup()

# Run Server
if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
