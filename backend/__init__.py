"""
Flask Application Factory
Creates and configures the Flask application with all routes and services
"""

from flask import Flask
from flask_cors import CORS


def create_app(config_name="development"):
    """
    Application factory pattern
    Creates a Flask app with all blueprints registered

    Args:
        config_name: 'development' or 'production'

    Returns:
        Configured Flask application
    """

    # Create Flask app
    app = Flask(__name__)

    # Configuration
    if config_name == "production":
        app.config["ENV"] = "production"
        app.config["DEBUG"] = False
        app.config["TESTING"] = False
    else:
        app.config["ENV"] = "development"
        app.config["DEBUG"] = True

    # Enable CORS for frontend communication
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": ["http://localhost:3000", "http://localhost:5173"],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type"],
            }
        },
    )

    # Register blueprints (routes)
    register_blueprints(app)

    # Add health check endpoint
    @app.route("/health", methods=["GET"])
    def health_check():
        return {"status": "ok", "service": "mental-health-framework"}, 200

    return app


def register_blueprints(app):
    """
    Register all route blueprints with the app
    Keeps app initialization clean and modular
    """

    # Import route blueprints
    from app.routes.analysis_routes import bp as analysis_bp
    from app.routes.recommendation_routes import bp as recommendation_bp

    # Register blueprints
    app.register_blueprint(analysis_bp)
    app.register_blueprint(recommendation_bp)

    print("✓ Registered blueprints:")
    print(f"  - {analysis_bp.name}: {analysis_bp.url_prefix}")
    print(f"  - {recommendation_bp.name}: {recommendation_bp.url_prefix}")
