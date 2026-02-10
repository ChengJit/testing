"""Detection modules for person, face, and box detection."""

from .person import PersonDetector
from .face import FaceRecognizer
from .box import BoxDetector
from .body import BodyRecognizer, BodyMatchResult

__all__ = ["PersonDetector", "FaceRecognizer", "BoxDetector", "BodyRecognizer", "BodyMatchResult"]
