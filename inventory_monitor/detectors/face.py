"""
Face recognition using InsightFace optimized for Jetson.
Supports automatic identity registration and locking.
Uses per-image embedding cache to avoid reprocessing known images.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import time

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class FaceMatch:
    """Face recognition result."""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # Face bounding box
    embedding: Optional[np.ndarray] = None


@dataclass
class UnknownFaceData:
    """Data for unknown face being collected for registration."""
    embeddings: List[np.ndarray] = field(default_factory=list)
    images: List[np.ndarray] = field(default_factory=list)
    last_seen: float = 0.0
    collection_count: int = 0


class FaceRecognizer:
    """
    InsightFace-based face recognition system.

    Features:
    - Efficient buffalo_s model for Jetson
    - Automatic unknown face collection
    - Identity locking after high-confidence match
    - Embedding persistence with per-image cache for fast retraining
    """

    def __init__(
        self,
        model_name: str = "buffalo_s",
        det_size: Tuple[int, int] = (320, 320),
        recognition_threshold: float = 0.45,
        lock_threshold: float = 0.75,
        embeddings_path: str = "known_embeddings.npz",
        faces_dir: str = "known_faces",
        use_gpu: bool = True,
        samples_for_registration: int = 5,
        sample_interval: float = 0.3,
    ):
        self.det_size = det_size
        self.recognition_threshold = recognition_threshold
        self.lock_threshold = lock_threshold
        self.embeddings_path = Path(embeddings_path)
        self.faces_dir = Path(faces_dir)
        self.samples_for_registration = samples_for_registration
        self.sample_interval = sample_interval

        # Per-image embedding cache path (sits next to the main embeddings file)
        self._cache_path = self.embeddings_path.with_name("embedding_cache.json")

        # Known faces storage
        self.known_names: List[str] = []
        self.known_embeddings: Optional[np.ndarray] = None

        # Per-image cache: { "person/filename.jpg": [512 floats] }
        self._image_cache: Dict[str, List[float]] = {}

        # Unknown face collection
        self.unknown_faces: Dict[int, UnknownFaceData] = defaultdict(UnknownFaceData)

        # Initialize model
        self._load_model(model_name, use_gpu)
        self._load_cache()
        self._load_embeddings()

        # Ensure faces directory exists
        self.faces_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self, model_name: str, use_gpu: bool):
        """Load InsightFace model."""
        try:
            from insightface.app import FaceAnalysis

            providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

            self.app = FaceAnalysis(
                name=model_name,
                providers=providers,
            )
            self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=self.det_size)

            logger.info(f"Face recognizer initialized with {model_name}")

        except ImportError as e:
            logger.error(f"Failed to import insightface: {e}")
            self.app = None
        except Exception as e:
            logger.error(f"Failed to initialize face recognition: {e}")
            self.app = None

    def _load_cache(self):
        """Load per-image embedding cache."""
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "r") as f:
                    self._image_cache = json.load(f)
                logger.info(f"Loaded embedding cache: {len(self._image_cache)} images")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self._image_cache = {}

    def _save_cache(self):
        """Save per-image embedding cache."""
        try:
            with open(self._cache_path, "w") as f:
                json.dump(self._image_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    def _image_cache_key(self, img_path: Path) -> str:
        """Cache key: relative path from faces_dir."""
        try:
            return str(img_path.relative_to(self.faces_dir))
        except ValueError:
            return str(img_path)

    def _load_embeddings(self):
        """Load known face embeddings from file."""
        if self.embeddings_path.exists():
            try:
                data = np.load(str(self.embeddings_path), allow_pickle=True)
                self.known_names = list(data["names"])
                self.known_embeddings = data["embeddings"]
                logger.info(f"Loaded {len(self.known_names)} known faces")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
                self.known_names = []
                self.known_embeddings = None
        else:
            logger.info("No existing embeddings found, will train from images")
            self.train_from_images()

    def _get_embedding_for_image(self, img_path: Path) -> Optional[np.ndarray]:
        """Get embedding for an image, using cache if available."""
        key = self._image_cache_key(img_path)

        # Check cache first
        if key in self._image_cache:
            return np.array(self._image_cache[key], dtype=np.float32)

        # Not cached — run InsightFace
        if self.app is None:
            return None

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None

            faces = self.app.get(img)
            if not faces:
                return None

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if face.normed_embedding is None:
                return None

            emb = face.normed_embedding
            # Store in cache
            self._image_cache[key] = emb.tolist()
            return emb

        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
            return None

    def train_from_images(self) -> int:
        """
        Train face recognition from images in known_faces directory.
        Uses per-image cache so only new images need InsightFace processing.

        Returns:
            Number of people trained
        """
        if self.app is None:
            logger.warning("Face model not loaded, cannot train")
            return 0

        if not self.faces_dir.exists():
            logger.info(f"Faces directory not found: {self.faces_dir}")
            return 0

        trained_count = 0
        new_names = []
        new_embeddings = []
        cache_dirty = False

        for person_dir in self.faces_dir.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            person_embeddings = []

            image_files = list(person_dir.glob("*.jpg")) + \
                          list(person_dir.glob("*.jpeg")) + \
                          list(person_dir.glob("*.png"))

            new_count = 0
            cached_count = 0

            for img_path in image_files:
                key = self._image_cache_key(img_path)
                was_cached = key in self._image_cache

                emb = self._get_embedding_for_image(img_path)
                if emb is not None:
                    person_embeddings.append(emb)
                    if was_cached:
                        cached_count += 1
                    else:
                        new_count += 1
                        cache_dirty = True

            if person_embeddings:
                avg_embedding = np.mean(person_embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

                new_names.append(person_name)
                new_embeddings.append(avg_embedding)
                trained_count += 1

                logger.info(
                    f"Trained {person_name}: {len(person_embeddings)} imgs "
                    f"({cached_count} cached, {new_count} new)")

        # Replace known faces entirely
        if new_embeddings:
            self.known_embeddings = np.array(new_embeddings)
            self.known_names = new_names
            self.save_embeddings()
            logger.info(f"Training complete: {trained_count} people")

        # Persist cache if we computed new embeddings
        if cache_dirty:
            self._save_cache()

        # Clean stale cache entries (images that were deleted)
        self._clean_cache()

        return trained_count

    def retrain_all(self) -> int:
        """Retrain all faces. Uses cache so only new images are processed."""
        self.known_names = []
        self.known_embeddings = None
        return self.train_from_images()

    def _clean_cache(self):
        """Remove cache entries for images that no longer exist."""
        to_remove = []
        for key in self._image_cache:
            full_path = self.faces_dir / key
            if not full_path.exists():
                to_remove.append(key)

        if to_remove:
            for key in to_remove:
                del self._image_cache[key]
            self._save_cache()
            logger.info(f"Cleaned {len(to_remove)} stale cache entries")

    def save_embeddings(self):
        """Save known face embeddings to file."""
        if self.known_embeddings is not None and len(self.known_names) > 0:
            np.savez(
                str(self.embeddings_path),
                names=np.array(self.known_names),
                embeddings=self.known_embeddings
            )
            logger.info(f"Saved {len(self.known_names)} face embeddings")

    def recognize(
        self,
        frame: np.ndarray,
        person_bbox: Optional[Tuple[int, int, int, int]] = None,
        track_id: Optional[int] = None
    ) -> Optional[FaceMatch]:
        """
        Recognize face in frame or person region.

        Args:
            frame: Full BGR frame
            person_bbox: Optional person bounding box to crop
            track_id: Optional track ID for unknown face collection

        Returns:
            FaceMatch if face found, None otherwise
        """
        if self.app is None:
            return None

        # Crop to person region if provided
        if person_bbox:
            x1, y1, x2, y2 = person_bbox
            h, w = frame.shape[:2]
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            roi = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            roi = frame
            offset = (0, 0)

        try:
            faces = self.app.get(roi)

            if not faces:
                return None

            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            embedding = face.normed_embedding

            if embedding is None:
                return None

            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            face_bbox = (
                fx1 + offset[0],
                fy1 + offset[1],
                fx2 + offset[0],
                fy2 + offset[1]
            )

            # Match against known faces
            if self.known_embeddings is not None and len(self.known_names) > 0:
                similarities = np.dot(self.known_embeddings, embedding)
                best_idx = np.argmax(similarities)
                best_score = similarities[best_idx]

                if best_score >= self.recognition_threshold:
                    return FaceMatch(
                        name=self.known_names[best_idx],
                        confidence=float(best_score),
                        bbox=face_bbox,
                        embedding=embedding
                    )

            # Unknown face - collect for potential registration
            if track_id is not None:
                self._collect_unknown_face(track_id, embedding, roi)

            return FaceMatch(
                name="Unknown",
                confidence=0.0,
                bbox=face_bbox,
                embedding=embedding
            )

        except Exception as e:
            logger.error(f"Face recognition error: {e}")
            return None

    def _collect_unknown_face(
        self,
        track_id: int,
        embedding: np.ndarray,
        face_image: np.ndarray
    ):
        """Collect unknown face samples for registration."""
        now = time.time()
        data = self.unknown_faces[track_id]

        if now - data.last_seen < self.sample_interval:
            return

        data.embeddings.append(embedding)
        data.images.append(face_image.copy())
        data.last_seen = now
        data.collection_count += 1

        logger.debug(
            f"Collected sample {data.collection_count}/{self.samples_for_registration} "
            f"for track {track_id}"
        )

    def get_registration_ready(self, track_id: int) -> bool:
        """Check if enough samples collected for registration."""
        if track_id not in self.unknown_faces:
            return False
        return len(self.unknown_faces[track_id].embeddings) >= self.samples_for_registration

    def register_face(self, track_id: int, name: str) -> bool:
        """
        Register a new face with collected samples.

        Args:
            track_id: Track ID with collected samples
            name: Name to assign to this face

        Returns:
            True if registration successful
        """
        if track_id not in self.unknown_faces:
            logger.warning(f"No samples for track {track_id}")
            return False

        data = self.unknown_faces[track_id]

        if len(data.embeddings) < 2:
            logger.warning(f"Not enough samples for track {track_id}")
            return False

        try:
            avg_embedding = np.mean(data.embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

            if self.known_embeddings is None:
                self.known_embeddings = avg_embedding.reshape(1, -1)
            else:
                self.known_embeddings = np.vstack([
                    self.known_embeddings,
                    avg_embedding.reshape(1, -1)
                ])
            self.known_names.append(name)

            person_dir = self.faces_dir / name
            person_dir.mkdir(parents=True, exist_ok=True)

            for i, img in enumerate(data.images):
                img_path = person_dir / f"{name}_{int(time.time())}_{i}.jpg"
                cv2.imwrite(str(img_path), img)

            self.save_embeddings()

            del self.unknown_faces[track_id]

            logger.info(f"Registered new face: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register face: {e}")
            return False

    def cleanup_stale_tracks(self, active_track_ids: List[int], max_age: float = 30.0):
        """Remove unknown face data for stale tracks."""
        now = time.time()
        stale_ids = []

        for track_id, data in self.unknown_faces.items():
            if track_id not in active_track_ids and now - data.last_seen > max_age:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            del self.unknown_faces[track_id]
