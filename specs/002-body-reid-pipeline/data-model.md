# Data Model: Body Re-Identification Pipeline

**Feature**: 002-body-reid-pipeline
**Date**: 2026-02-10

## Entities

### PersonProfile

Complete identity record combining face and body recognition data.

| Field | Type | Description |
|-------|------|-------------|
| name | str | Display name / identifier |
| face_embedding | Optional[ndarray] | 512D InsightFace embedding |
| body_embedding | Optional[ndarray] | 128D OSNet body embedding |
| color_histogram | Optional[ndarray] | 1152D HSV color histogram |
| front_view_crop | Optional[bytes] | JPEG of front view (face visible) |
| back_view_crop | Optional[bytes] | JPEG of back view |
| registration_timestamp | str | ISO timestamp of first registration |
| last_seen | str | ISO timestamp of last detection |
| sample_count | int | Number of samples collected |
| auto_registered | bool | Whether auto-registered (vs manual) |

**Storage**: `person_profiles.json` + `body_embeddings.npz`

**Relationships**:
- One PersonProfile has 0-1 face_embedding
- One PersonProfile has 0-N body_embeddings (averaged)
- One PersonProfile links to multiple CaptureEvents

---

### CaptureEvent

Queued item for the recognition thread to process.

| Field | Type | Description |
|-------|------|-------------|
| track_id | int | ByteTracker track ID |
| timestamp | str | ISO timestamp of capture |
| event_type | str | "entry" or "exit" |
| full_body_crop | ndarray | Full body image from YOLO bbox |
| face_crop | Optional[ndarray] | Face ROI if detected |
| face_detected | bool | Whether face was visible |
| box_count | int | Number of boxes carried |
| processing_status | str | "pending", "processing", "resolved" |
| door_crossing_logged | bool | Whether initial event was logged |

**Lifecycle**:
1. Created when person crosses door line
2. Queued for recognition thread
3. Status updated as recognition progresses
4. Deleted after IDENTITY_RESOLVED sent (or timeout)

---

### BodyEmbedding

Feature vector for body appearance matching.

| Field | Type | Description |
|-------|------|-------------|
| embedding | ndarray | 128D OSNet feature vector (L2 normalized) |
| histogram | ndarray | 1152D HSV color histogram |
| view_type | str | "front", "back", or "side" |
| clothing_descriptor | dict | Dominant colors, patterns |
| extraction_timestamp | str | When embedding was extracted |

**Matching**:
- Cosine similarity for embedding comparison
- Chi-squared distance for histogram comparison
- Combined score with configurable weights

---

### RecognitionResult

Output from the recognition thread.

| Field | Type | Description |
|-------|------|-------------|
| track_id | int | Original track ID |
| identity | Optional[str] | Matched person name or None |
| confidence | float | Match confidence 0.0-1.0 |
| match_method | str | "face", "body", "closest", "none" |
| face_score | Optional[float] | Face recognition score if attempted |
| body_score | Optional[float] | Body recognition score if attempted |
| needs_review | bool | Flag for low-margin matches |
| alternative_matches | List[Tuple[str, float]] | Top-3 candidates if ambiguous |

**State Transitions**:
```
pending → face_matched (confidence > 0.75)
pending → body_matched (face failed, body > 0.7)
pending → closest_matched (body failed, closest > 0.5)
pending → unknown (no match above thresholds)
```

---

### RecognitionQueue

Thread-safe queue between capture and recognition threads.

| Field | Type | Description |
|-------|------|-------------|
| events | Queue[CaptureEvent] | Pending capture events |
| max_size | int | Maximum queue depth (default: 20) |
| drop_oldest | bool | Drop oldest when full (True) |

**Behavior**:
- Capture thread adds events
- Recognition thread pops events FIFO
- If queue full, oldest event dropped (with warning log)

---

## Storage Schema

### body_embeddings.npz

```python
{
    'names': ndarray,        # Shape: (N,), dtype=object (strings)
    'embeddings': ndarray,   # Shape: (N, 128), dtype=float32
    'histograms': ndarray,   # Shape: (N, 1152), dtype=float32
    'view_types': ndarray,   # Shape: (N,), dtype=object
}
```

### person_profiles.json

```json
{
  "Person_001": {
    "registration_timestamp": "2026-02-10T10:30:00",
    "last_seen": "2026-02-10T15:45:00",
    "sample_count": 5,
    "auto_registered": true,
    "has_face": true,
    "has_body_front": true,
    "has_body_back": false
  }
}
```

---

## Validation Rules

1. **PersonProfile**:
   - name MUST be unique
   - At least one of face_embedding or body_embedding MUST be present
   - sample_count MUST be >= 1

2. **CaptureEvent**:
   - track_id MUST be positive integer
   - event_type MUST be "entry" or "exit"
   - full_body_crop MUST be non-empty ndarray

3. **RecognitionResult**:
   - confidence MUST be in [0.0, 1.0]
   - match_method MUST be one of: "face", "body", "closest", "none"
   - If identity is set, confidence MUST be > 0

4. **Embeddings**:
   - Body embeddings MUST be L2-normalized (unit vectors)
   - Histograms MUST sum to 1.0 (normalized)
