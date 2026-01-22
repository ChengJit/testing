"""
Face recognition using InsightFace optimized for Jetson.
Supports automatic identity registration and locking.
"""

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
    - Embedding persistence
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

        # Known faces storage
        self.known_names: List[str] = []
        self.known_embeddings: Optional[np.ndarray] = None

        # Unknown face collection
        self.unknown_faces: Dict[int, UnknownFaceData] = defaultdict(UnknownFaceData)

        # Initialize model
        self._load_model(model_name, use_gpu)
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
            logger.info("No existing embeddings found, starting fresh")

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
            # Add margin
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
            # Detect and analyze faces
            faces = self.app.get(roi)

            if not faces:
                return None

            # Use largest face (closest to camera)
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            # Get embedding
            embedding = face.normed_embedding

            if embedding is None:
                return None

            # Adjust face bbox to frame coordinates
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            face_bbox = (
                fx1 + offset[0],
                fy1 + offset[1],
                fx2 + offset[0],
                fy2 + offset[1]
            )

            # Match against known faces
            if self.known_embeddings is not None and len(self.known_names) > 0:
                # Cosine similarity (embeddings are normalized)
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

        # Check collection interval
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
            # Average embeddings for robustness
            avg_embedding = np.mean(data.embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

            # Add to known faces
            if self.known_embeddings is None:
                self.known_embeddings = avg_embedding.reshape(1, -1)
            else:
                self.known_embeddings = np.vstack([
                    self.known_embeddings,
                    avg_embedding.reshape(1, -1)
                ])
            self.known_names.append(name)

            # Save face images
            person_dir = self.faces_dir / name
            person_dir.mkdir(parents=True, exist_ok=True)

            for i, img in enumerate(data.images):
                img_path = person_dir / f"{name}_{int(time.time())}_{i}.jpg"
                cv2.imwrite(str(img_path), img)

            # Save embeddings
            self.save_embeddings()

            # Clear collected data
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
