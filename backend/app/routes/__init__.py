"""
Routes Package
Contains all API endpoint blueprints

Each route file defines a Flask Blueprint with related endpoints:
- analysis_routes: Depression classification endpoints
- recommendation_routes: Resource recommendation endpoints
"""

# Import blueprints
from .analysis_routes import bp as analysis_bp
from .recommendation_routes import bp as recommendation_bp

# Export blueprints
__all__ = ["analysis_bp", "recommendation_bp"]


# List all registered routes
def list_routes():
    """
    Helper function to list all available routes
    Useful for debugging
    """
    routes = {
        "analysis": ["POST /api/analysis/predict", "POST /api/analysis/batch-predict"],
        "recommendations": ["POST /api/recommendations/get-resources"],
    }
    return routes


print("✓ Routes package loaded")
print(f"  Available endpoints: {sum(len(v) for v in list_routes().values())}")
