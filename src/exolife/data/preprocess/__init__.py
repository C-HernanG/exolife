"""Preprocess module for data preprocessing operations."""

# Import concrete implementations
from .base_preprocessor import BasePreprocessor, PreprocessorConfig, PreprocessorResult
from .registry import PreprocessorRegistry


# Convenience functions
def get_preprocessor_info():
    """
    Get information about available preprocessors.
    """
    return PreprocessorRegistry.get_preprocessor_info()


__all__ = [
    "BasePreprocessor",
    "PreprocessorConfig",
    "PreprocessorResult",
    "PreprocessorRegistry",
    "get_preprocessor_info",
]
