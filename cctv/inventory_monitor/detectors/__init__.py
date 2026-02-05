"""Detection modules for person, face, and box detection."""

from .person import PersonDetector
from .face import FaceRecognizer
from .box import BoxDetector

__all__ = ["PersonDetector", "FaceRecognizer", "BoxDetector"]
