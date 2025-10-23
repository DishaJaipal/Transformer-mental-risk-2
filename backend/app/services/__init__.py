"""
Services Package
Exports all reusable business logic services

Services are singleton instances that provide:
- Preprocessing: Text cleaning
- Embedding: Feature extraction
- Classification: Depression prediction
- Recommendations: Resource suggestions
"""

# Import all services
from .preprocessor import preprocessing_service
from .embedding_service import embedding_service
from .classifier_service import classifier_service
from .recommendation_service import recommendation_service

# Export services for easy import
__all__ = [
    "preprocessing_service",
    "embedding_service",
    "classifier_service",
    "recommendation_service",
]

# Version info
__version__ = "1.0.0"

print(f"✓ Services package loaded (v{__version__})")
