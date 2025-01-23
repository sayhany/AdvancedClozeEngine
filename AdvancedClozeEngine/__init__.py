"""AttentionFlash: A next-generation cloze flashcard system using BERT attention.

This package provides tools for generating intelligent cloze-style flashcards
using BERT's attention mechanisms. It can automatically identify important
concepts and create effective learning materials.

Example:
    >>> from attention_flash import create_engine
    >>> engine = create_engine()
    >>> result = engine.generate_cloze("The mitochondria is the powerhouse of the cell.")
    >>> print(result["masked"])
"""

from typing import Optional, List, Union
from pathlib import Path

from .engine import AdvancedClozeEngine

def create_engine(
    model_name: str = "bert-base-uncased",
    domain_tokens: Optional[List[str]] = None,
    device: Optional[str] = None
) -> AdvancedClozeEngine:
    """Create a new AdvancedClozeEngine instance with the specified configuration.

    Args:
        model_name: Name of the pretrained BERT model to use
        domain_tokens: Optional list of domain-specific terms to boost
        device: Optional device specification ('cuda' or 'cpu')

    Returns:
        Configured AdvancedClozeEngine instance
    """
    return AdvancedClozeEngine(
        model_name=model_name,
        domain_tokens=domain_tokens,
        device=device
    )

def get_package_root() -> Path:
    """Get the root directory of the AdvancedClozeEngine package."""
    return Path(__file__).parent

__version__ = "0.1.0"
__author__ = "Sayhan Yalvacer"
__license__ = "MIT"
__description__ = "Next-generation cloze flashcard system using BERT attention"

__all__ = [
    "AdvancedClozeEngine",
    "create_engine",
    "get_package_root",
    "__version__",
    "__author__",
    "__license__",
    "__description__"
]