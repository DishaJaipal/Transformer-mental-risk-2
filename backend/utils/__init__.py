"""
Utilities Package
Reusable utility functions and classes

Contains:
- text_preprocessing: RedditPreprocessor for text cleaning
- (future) data_loaders: Dataset loading utilities
- (future) validators: Input validation functions
"""

# Import main utilities
from backend.utils.textPreprocessor import RedditPreprocessor

# Export for easy import
__all__ = ["RedditPreprocessor"]

__version__ = "1.0.0"

print(f"✓ Utils package loaded (v{__version__})")
