from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from .models import Base, engine
from .routes import api_bp
from .ml.model_helper import load_ml_assets
import os

def create_app() -> Flask:
    app = Flask(__name__)

    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Register API routes
    app.register_blueprint(api_bp, url_prefix="/api")

    # Load ML assets
    load_ml_assets()

    # Dynamic frontend path — relative to this file, works in dev and on VPS
    _here = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_here))
    FRONTEND_PATH = os.path.join(_project_root, "frontend")
    if not os.path.exists(FRONTEND_PATH):
        FRONTEND_PATH = os.path.join(os.getcwd(), "frontend")

    # Serve frontend
    @app.route("/")
    def index():
        return send_from_directory(FRONTEND_PATH, 'index.html')

    @app.route("/<path:filename>")
    def serve_static(filename):
        return send_from_directory(FRONTEND_PATH, filename)

    # API info endpoint
    @app.route("/api")
    def api_info():
        return jsonify({
            "name": "VeriCheck API",
            "status": "running",
            "version": "1.0.0",
            "free_tier_limit": 200
        })

    return app

app = create_app()