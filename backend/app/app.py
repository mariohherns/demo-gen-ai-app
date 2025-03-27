from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os
import sys

#  Dynamically set the project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from services.fracture_model import FractureModel
from services.openai_client import OpenAIService
from routes.full_analyze import register_analyze_routes

load_dotenv()

# Init Flask
app = Flask(__name__)
CORS(app)

# fracture_model = FractureModel("models/fracture_detector.pth")
openai_service = OpenAIService(os.getenv("OPENAI_API_KEY"))


# Instantiate services
fracture_model = FractureModel("backend/models/fracture_detector.pth")
print(fracture_model, "CHECK MODEL ***")
# Register routes
register_analyze_routes(app, fracture_model, openai_service)

if __name__ == "__main__":
    app.run(debug=True)