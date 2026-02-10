"""
Body Re-Identification Module.

Uses hybrid approach combining:
1. Color histogram (HSV) for quick matching
2. OSNet embeddings for accurate deep matching

Designed for Jetson Orin Nano with TensorRT optimization.
"""

import logging
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# OSNet dimensions
OSNET_INPUT_SIZE = (256, 128)  # Height x Width
OSNET_EMBEDDING_DIM = 512  # OSNet output dimension
HISTOGRAM_DIM = 1152  # 18 * 8 * 8 bins


@dataclass
class BodyMatchResult:
    """Result from body matching."""
    name: Optional[str] = None
    confidence: float = 0.0
    histogram_score: float = 0.0
    embedding_score: float = 0.0
    needs_review: bool = False
    margin: float = 0.0  # Difference from second-best match


class BodyRecognizer:
    """
    Body re-identification using color histograms and OSNet embeddings.

    Matches people by their body appearance (clothing, body shape) when
    face is not visible.
    """

    def __init__(
        self,
        embeddings_file: str = "data/body_embeddings.npz",
        profiles_file: str = "data/person_profiles.json",
        body_model_path: Optional[str] = None,
        use_osnet: bool = True,
        use_histogram: bool = True,
        histogram_weight: float = 0.5,
        threshold: float = 0.6,
        closest_threshold: float = 0.5,
        margin_threshold: float = 0.1,
    ):
        """
        Initialize body recognizer.

        Args:
            embeddings_file: Path to NPZ file storing body embeddings
            profiles_file: Path to JSON file storing person profiles
            body_model_path: Path to OSNet model file
            use_osnet: Whether to use OSNet for deep matching
            use_histogram: Whether to use color histogram matching
            histogram_weight: Weight for histogram score (0-1)
            threshold: Minimum confidence for positive match
            closest_threshold: Threshold for closest-match fallback
            margin_threshold: Flag for review if top-2 margin < this
        """
        self.embeddings_file = Path(embeddings_file)
        self.profiles_file = Path(profiles_file)
        self.body_model_path = body_model_path
        self.use_osnet = use_osnet
        self.use_histogram = use_histogram
        self.histogram_weight = histogram_weight
        self.osnet_weight = 1.0 - histogram_weight
        self.threshold = threshold
        self.closest_threshold = closest_threshold
        self.margin_threshold = margin_threshold

        # Known embeddings: name -> embedding data
        self.known_names: List[str] = []
        self.known_embeddings: Optional[np.ndarray] = None  # Shape: (N, OSNET_EMBEDDING_DIM)
        self.known_histograms: Optional[np.ndarray] = None  # Shape: (N, HISTOGRAM_DIM)
        self.known_view_types: List[str] = []  # "front", "back", "side"

        # OSNet model (loaded lazily)
        self._osnet_model = None
        self._osnet_available = False

        # Load existing embeddings
        self._load_embeddings()

        # Try to load OSNet model
        if self.use_osnet:
            self._load_osnet_model()

        logger.info(
            f"BodyRecognizer initialized: {len(self.known_names)} known persons, "
            f"OSNet={'available' if self._osnet_available else 'unavailable'}"
        )

    def _load_embeddings(self):
        """Load embeddings from NPZ file."""
        if not self.embeddings_file.exists():
            logger.info(f"No body embeddings file found at {self.embeddings_file}")
            return

        try:
            data = np.load(str(self.embeddings_file), allow_pickle=True)
            self.known_names = list(data['names'])
            self.known_embeddings = data.get('embeddings')
            self.known_histograms = data.get('histograms')
            self.known_view_types = list(data.get('view_types', []))

            if len(self.known_view_types) != len(self.known_names):
                self.known_view_types = ['unknown'] * len(self.known_names)

            logger.info(f"Loaded {len(self.known_names)} body embeddings")
        except Exception as e:
            logger.error(f"Failed to load body embeddings: {e}")

    def save_embeddings(self):
        """Save embeddings to NPZ file."""
        self.embeddings_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'names': np.array(self.known_names, dtype=object),
            'view_types': np.array(self.known_view_types, dtype=object),
        }

        if self.known_embeddings is not None:
            data['embeddings'] = self.known_embeddings
        if self.known_histograms is not None:
            data['histograms'] = self.known_histograms

        np.savez(str(self.embeddings_file), **data)
        logger.debug(f"Saved {len(self.known_names)} body embeddings")

    def _load_osnet_model(self):
        """Load OSNet model for body embedding extraction."""
        try:
            import torch

            if self.body_model_path and Path(self.body_model_path).exists():
                # Load from specified path
                self._osnet_model = torch.load(self.body_model_path, map_location='cpu')
                self._osnet_model.eval()
                self._osnet_available = True
                logger.info(f"Loaded OSNet model from {self.body_model_path}")
            else:
                # OSNet not available - will use histogram only
                logger.info("OSNet model not found, using histogram-only matching")
                self._osnet_available = False
        except ImportError:
            logger.warning("PyTorch not available, using histogram-only matching")
            self._osnet_available = False
        except Exception as e:
            logger.warning(f"Failed to load OSNet model: {e}, using histogram-only")
            self._osnet_available = False

    def extract_color_histogram(self, body_crop: np.ndarray) -> np.ndarray:
        """
        Extract HSV color histogram from body crop.

        Args:
            body_crop: BGR image of body region

        Returns:
            Normalized histogram vector (1152D)
        """
        if body_crop is None or body_crop.size == 0:
            return np.zeros(HISTOGRAM_DIM, dtype=np.float32)

        # Convert to HSV
        hsv = cv2.cvtColor(body_crop, cv2.COLOR_BGR2HSV)

        # Calculate histogram: 18 H bins, 8 S bins, 8 V bins
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [18, 8, 8],
            [0, 180, 0, 256, 0, 256]
        )

        # Normalize and flatten
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)

    def histogram_match(
        self,
        query_hist: np.ndarray,
        known_hist: np.ndarray
    ) -> float:
        """
        Match histograms using Chi-squared distance.

        Args:
            query_hist: Query histogram
            known_hist: Known histogram to compare against

        Returns:
            Similarity score (0-1, higher = better match)
        """
        if query_hist is None or known_hist is None:
            return 0.0

        # Chi-squared distance
        distance = cv2.compareHist(
            query_hist.astype(np.float32),
            known_hist.astype(np.float32),
            cv2.HISTCMP_CHISQR
        )

        # Convert distance to similarity (0-1)
        similarity = 1.0 / (1.0 + distance)
        return float(similarity)

    def extract_body_embedding(self, body_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract OSNet embedding from body crop.

        Args:
            body_crop: BGR image of body region

        Returns:
            L2-normalized embedding vector (512D) or None if OSNet unavailable
        """
        if not self._osnet_available or self._osnet_model is None:
            return None

        if body_crop is None or body_crop.size == 0:
            return None

        try:
            import torch
            import torch.nn.functional as F

            # Preprocess: resize to OSNet input size
            img = cv2.resize(body_crop, (OSNET_INPUT_SIZE[1], OSNET_INPUT_SIZE[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize (ImageNet stats)
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std

            # Convert to tensor: (H, W, C) -> (1, C, H, W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

            # Extract embedding
            with torch.no_grad():
                embedding = self._osnet_model(img_tensor)

            # L2 normalize
            embedding = F.normalize(embedding, p=2, dim=1)

            return embedding.cpu().numpy().flatten()

        except Exception as e:
            logger.debug(f"OSNet embedding extraction failed: {e}")
            return None

    def combined_score(
        self,
        histogram_score: float,
        embedding_score: float
    ) -> float:
        """
        Combine histogram and embedding scores.

        Args:
            histogram_score: Score from histogram matching (0-1)
            embedding_score: Score from OSNet matching (0-1)

        Returns:
            Combined score (0-1)
        """
        if not self._osnet_available or embedding_score == 0:
            # OSNet not available, use histogram only
            return histogram_score

        return (self.histogram_weight * histogram_score +
                self.osnet_weight * embedding_score)

    def recognize(self, body_crop: np.ndarray) -> BodyMatchResult:
        """
        Recognize a person by their body appearance.

        Args:
            body_crop: BGR image of body region

        Returns:
            BodyMatchResult with best match (or None if no match)
        """
        if len(self.known_names) == 0:
            return BodyMatchResult()

        # Extract features
        query_hist = self.extract_color_histogram(body_crop)
        query_embed = self.extract_body_embedding(body_crop)

        # Calculate scores for all known persons
        scores = []
        for i, name in enumerate(self.known_names):
            hist_score = 0.0
            embed_score = 0.0

            if self.use_histogram and self.known_histograms is not None:
                hist_score = self.histogram_match(query_hist, self.known_histograms[i])

            if self.use_osnet and query_embed is not None and self.known_embeddings is not None:
                # Cosine similarity (embeddings are L2 normalized)
                embed_score = float(np.dot(query_embed, self.known_embeddings[i]))
                # Normalize to 0-1 range
                embed_score = (embed_score + 1) / 2

            combined = self.combined_score(hist_score, embed_score)
            scores.append((name, combined, hist_score, embed_score))

        # Sort by combined score
        scores.sort(key=lambda x: x[1], reverse=True)

        if len(scores) == 0:
            return BodyMatchResult()

        best_name, best_score, best_hist, best_embed = scores[0]

        # Calculate margin from second-best
        margin = 1.0
        if len(scores) > 1:
            margin = best_score - scores[1][1]

        # Check if match is good enough
        if best_score >= self.threshold:
            return BodyMatchResult(
                name=best_name,
                confidence=best_score,
                histogram_score=best_hist,
                embedding_score=best_embed,
                needs_review=(margin < self.margin_threshold),
                margin=margin
            )

        return BodyMatchResult(
            confidence=best_score,
            histogram_score=best_hist,
            embedding_score=best_embed,
            margin=margin
        )

    def find_closest(self, body_crop: np.ndarray) -> BodyMatchResult:
        """
        Find the closest matching person, even if below threshold.

        Args:
            body_crop: BGR image of body region

        Returns:
            BodyMatchResult with closest match
        """
        if len(self.known_names) == 0:
            return BodyMatchResult()

        result = self.recognize(body_crop)

        # If we got a match above threshold, return it
        if result.name is not None:
            return result

        # Otherwise, check if closest match is above the lower threshold
        query_hist = self.extract_color_histogram(body_crop)
        query_embed = self.extract_body_embedding(body_crop)

        scores = []
        for i, name in enumerate(self.known_names):
            hist_score = 0.0
            embed_score = 0.0

            if self.use_histogram and self.known_histograms is not None:
                hist_score = self.histogram_match(query_hist, self.known_histograms[i])

            if self.use_osnet and query_embed is not None and self.known_embeddings is not None:
                embed_score = float(np.dot(query_embed, self.known_embeddings[i]))
                embed_score = (embed_score + 1) / 2

            combined = self.combined_score(hist_score, embed_score)
            scores.append((name, combined, hist_score, embed_score))

        scores.sort(key=lambda x: x[1], reverse=True)

        if len(scores) == 0:
            return BodyMatchResult()

        best_name, best_score, best_hist, best_embed = scores[0]

        margin = 1.0
        if len(scores) > 1:
            margin = best_score - scores[1][1]

        # Return closest match if above the lower threshold
        if best_score >= self.closest_threshold:
            return BodyMatchResult(
                name=best_name,
                confidence=best_score,
                histogram_score=best_hist,
                embedding_score=best_embed,
                needs_review=(margin < self.margin_threshold),
                margin=margin
            )

        return BodyMatchResult(
            confidence=best_score,
            histogram_score=best_hist,
            embedding_score=best_embed,
            margin=margin
        )

    def register_body(
        self,
        name: str,
        body_crop: np.ndarray,
        view_type: str = "unknown"
    ) -> bool:
        """
        Register a new person's body appearance.

        Args:
            name: Person's name/identifier
            body_crop: BGR image of body region
            view_type: "front", "back", or "side"

        Returns:
            True if registration successful
        """
        if body_crop is None or body_crop.size == 0:
            return False

        histogram = self.extract_color_histogram(body_crop)
        embedding = self.extract_body_embedding(body_crop)

        # Add to lists
        self.known_names.append(name)
        self.known_view_types.append(view_type)

        # Add histogram
        if self.known_histograms is None:
            self.known_histograms = histogram.reshape(1, -1)
        else:
            self.known_histograms = np.vstack([self.known_histograms, histogram])

        # Add embedding (if available)
        if embedding is not None:
            if self.known_embeddings is None:
                self.known_embeddings = embedding.reshape(1, -1)
            else:
                self.known_embeddings = np.vstack([self.known_embeddings, embedding])
        elif self.known_embeddings is not None:
            # Add zero embedding as placeholder
            self.known_embeddings = np.vstack([
                self.known_embeddings,
                np.zeros((1, self.known_embeddings.shape[1]))
            ])

        # Save to file
        self.save_embeddings()

        logger.info(f"Registered body for '{name}' (view_type={view_type})")
        return True

    def update_body_embedding(
        self,
        name: str,
        body_crop: np.ndarray,
        alpha: float = 0.3
    ) -> bool:
        """
        Update existing body embedding with new sample (running average).

        Args:
            name: Person's name
            body_crop: New body crop
            alpha: Weight for new sample (0-1)

        Returns:
            True if update successful
        """
        if name not in self.known_names:
            return False

        idx = self.known_names.index(name)

        # Update histogram
        new_hist = self.extract_color_histogram(body_crop)
        if self.known_histograms is not None:
            self.known_histograms[idx] = (
                (1 - alpha) * self.known_histograms[idx] +
                alpha * new_hist
            )
            # Re-normalize
            self.known_histograms[idx] /= np.sum(self.known_histograms[idx])

        # Update embedding
        new_embed = self.extract_body_embedding(body_crop)
        if new_embed is not None and self.known_embeddings is not None:
            self.known_embeddings[idx] = (
                (1 - alpha) * self.known_embeddings[idx] +
                alpha * new_embed
            )
            # Re-normalize
            self.known_embeddings[idx] /= np.linalg.norm(self.known_embeddings[idx])

        self.save_embeddings()
        logger.debug(f"Updated body embedding for '{name}'")
        return True

    def get_known_persons(self) -> List[str]:
        """Get list of known person names."""
        return list(set(self.known_names))

    def has_person(self, name: str) -> bool:
        """Check if a person is registered."""
        return name in self.known_names
