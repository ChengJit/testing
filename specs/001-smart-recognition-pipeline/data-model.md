# Data Model: Smart Recognition Pipeline

**Branch**: `001-smart-recognition-pipeline` | **Date**: 2026-02-05

## Entities

### DeferredEvent

Tracks entry/exit events that were logged before identity resolution
completed. Used to send follow-up identity-update events to the API.

| Field                | Type             | Description                                      |
|----------------------|------------------|--------------------------------------------------|
| track_id             | int              | Unique tracker ID for this person                |
| event_type           | EventType        | PERSON_ENTERED or PERSON_EXITED                  |
| timestamp            | str (ISO 8601)   | When the door crossing occurred                  |
| box_count            | int              | Number of boxes at time of crossing              |
| identity             | Optional[str]    | None initially; populated when resolved          |
| identity_confidence  | float            | 0.0 initially; updated on resolution             |
| resolved             | bool             | False until identity is assigned                 |
| resolved_timestamp   | Optional[str]    | When identity was resolved (ISO 8601)            |

**State transitions**:
```
Created (identity=None, resolved=False)
    → Resolved (identity=<name>, resolved=True, resolved_timestamp=now)
    → Expired (if track is removed without resolution)
```

**Storage**: In-memory dict keyed by `track_id`. No persistence needed
beyond the existing event log files — the `IDENTITY_RESOLVED` event
in JSONL/CSV serves as the correction record.

---

### AutoRegistration

Records automatically registered faces for operator review and
correction tracking.

| Field                | Type            | Description                                      |
|----------------------|-----------------|--------------------------------------------------|
| generated_label      | str             | Auto-generated name (e.g., "Person_001")         |
| track_id             | int             | Track that triggered auto-registration           |
| sample_count         | int             | Number of face samples collected                 |
| registration_time    | str (ISO 8601)  | When auto-registration occurred                  |
| reviewed             | bool            | Whether operator has reviewed/corrected          |
| corrected_label      | Optional[str]   | Operator-assigned name, if corrected             |
| corrected_time       | Optional[str]   | When correction was made (ISO 8601)              |

**Storage**: Append-only JSONL file at
`data/auto_registrations.jsonl`. Survives restarts. Loaded on startup
to determine the next auto-registration counter value.

---

### RecognitionPriority

Ephemeral per-frame priority scoring for face recognition scheduling.
Not persisted.

| Field                | Type            | Description                                      |
|----------------------|-----------------|--------------------------------------------------|
| track_id             | int             | Track being scored                               |
| distance_to_door     | float           | Absolute pixel distance from track center to door line |
| approaching          | bool            | True if track is moving toward the door          |
| priority_score       | float           | Computed score: higher = process first            |

**Computation**:
```
base = 1.0 / max(1, distance_to_door)
multiplier = 2.0 if approaching else 1.0
priority_score = base * multiplier
```

**Storage**: Computed in-memory per frame. Not persisted.

---

### VerifiedTrainingSample

Extension of the existing training manifest entry with auto-verification.

| Field                | Type            | Description                                      |
|----------------------|-----------------|--------------------------------------------------|
| path                 | str             | Image file path                                  |
| label                | str             | "has_box" or "no_box"                            |
| timestamp            | str (ISO 8601)  | When sample was captured                         |
| track_id             | int             | Tracker ID of the person                         |
| identity             | Optional[str]   | Person identity if known                         |
| box_count            | int             | Number of boxes detected                         |
| methods              | List[str]       | Detection methods used                           |
| confidences          | List[float]     | Confidence scores per detection                  |
| verified             | bool            | True if auto-verified or manually verified       |
| verification_method  | Optional[str]   | "auto_confidence" or "manual"                    |

**Auto-verification rule**: A sample is auto-verified when ALL box
detections in the sample have confidence >= the configured
`auto_verify_threshold` (default: 0.8) and the detection method is
`yolo_custom` (the trained model, not fallback methods).

**Storage**: Existing `manifest.jsonl` in the training output
directory. New fields (`verification_method`) are added to the schema.

---

## Relationship Map

```
TrackedObject (bytetrack.py)
    │
    ├── has → DeferredEvent (0 or 1 per entry/exit)
    │         Created when door crossed with identity=None
    │         Resolved when face recognition matches
    │
    ├── has → AutoRegistration (0 or 1 per track)
    │         Created when auto-register triggers
    │         Updated when operator corrects
    │
    ├── scored by → RecognitionPriority (1 per frame)
    │               Ephemeral, computed in AIWorker
    │
    └── produces → VerifiedTrainingSample (0..N)
                    Via TrainingDataCollector
```

## Config Additions

New fields to add to the existing `Config` dataclass hierarchy:

### RecognitionConfig (new section or extend DetectionConfig)

| Field                    | Type   | Default      | Description                            |
|--------------------------|--------|--------------|----------------------------------------|
| auto_register            | bool   | False        | Enable auto-registration of faces      |
| auto_register_pattern    | str    | "Person_{}"  | Naming pattern for auto-registered faces |
| priority_recognition     | bool   | True         | Enable door-proximity priority sorting |
| identity_resolve_timeout | float  | 30.0         | Seconds before marking deferred event as expired |

### TrainingConfig (extend existing)

| Field                    | Type   | Default      | Description                            |
|--------------------------|--------|--------------|----------------------------------------|
| auto_verify_threshold    | float  | 0.8          | Min confidence for auto-verification   |
| retrain_min_samples      | int    | 500          | Min verified samples before retraining |
