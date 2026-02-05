# API Event Contracts: Smart Recognition Pipeline

**Branch**: `001-smart-recognition-pipeline` | **Date**: 2026-02-05

## Existing Event Types (unchanged)

### POST /event — Entry Event

```json
{
  "timestamp": "2026-02-05T14:30:00.123456",
  "event_type": "entered",
  "track_id": 42,
  "identity": "John Doe",
  "identity_confidence": 0.85,
  "box_count": 2,
  "direction": "entering",
  "camera_id": "cam-001",
  "details": {},
  "image_base64": "<base64-encoded-jpeg>"
}
```

### POST /event — Exit Event

```json
{
  "timestamp": "2026-02-05T14:35:00.654321",
  "event_type": "exited",
  "track_id": 42,
  "identity": "John Doe",
  "identity_confidence": 0.85,
  "box_count": 1,
  "direction": "exiting",
  "camera_id": "cam-001",
  "details": {
    "entry_box_count": 2,
    "exit_box_count": 1,
    "box_difference": -1
  },
  "image_base64": "<base64-encoded-jpeg>"
}
```

## New Event Type

### POST /event — Identity Resolved Event

Sent when face recognition resolves an identity for a track that
previously had an entry/exit event logged with `identity=null`.

```json
{
  "timestamp": "2026-02-05T14:30:05.789012",
  "event_type": "identity_resolved",
  "track_id": 42,
  "identity": "John Doe",
  "identity_confidence": 0.85,
  "box_count": 0,
  "direction": null,
  "camera_id": "cam-001",
  "details": {
    "original_event_type": "entered",
    "original_timestamp": "2026-02-05T14:30:00.123456",
    "resolution_delay_ms": 5665
  }
}
```

**Fields**:

| Field                  | Type            | Required | Description                                    |
|------------------------|-----------------|----------|------------------------------------------------|
| timestamp              | str (ISO 8601)  | Yes      | When the identity was resolved                 |
| event_type             | str             | Yes      | Always `"identity_resolved"`                   |
| track_id               | int             | Yes      | Links to the original entry/exit event         |
| identity               | str             | Yes      | Resolved person name                           |
| identity_confidence    | float           | Yes      | Recognition confidence score                   |
| box_count              | int             | Yes      | 0 (not relevant for this event)                |
| direction              | null            | Yes      | null (not relevant for this event)             |
| camera_id              | str             | Yes      | Camera that produced the original event        |
| details.original_event_type | str        | Yes      | "entered" or "exited"                          |
| details.original_timestamp  | str        | Yes      | Timestamp of the original unresolved event     |
| details.resolution_delay_ms | int        | Yes      | Milliseconds between original event and resolution |

### POST /event — Auto-Registration Event (informational)

Sent when the system auto-registers a new face. Informational only —
the ops-portal may use this to display new personnel notifications.

```json
{
  "timestamp": "2026-02-05T14:32:00.000000",
  "event_type": "face_auto_registered",
  "track_id": 55,
  "identity": "Person_003",
  "identity_confidence": 0.0,
  "box_count": 0,
  "direction": null,
  "camera_id": "cam-001",
  "details": {
    "sample_count": 5,
    "auto_generated": true
  }
}
```

## Batch Endpoint (unchanged)

### POST /events/batch

```json
{
  "events": [
    { "...event1..." },
    { "...event2..." }
  ]
}
```

Response:
```json
{
  "successCount": 2,
  "failCount": 0
}
```

The batch endpoint accepts any mix of event types, including the new
`identity_resolved` and `face_auto_registered` types.

## Internal Event Types (Python enum extension)

```python
class EventType(Enum):
    PERSON_ENTERED = "entered"
    PERSON_EXITED = "exited"
    BOX_DETECTED = "box_detected"
    IDENTITY_RECOGNIZED = "identity_recognized"
    IDENTITY_RESOLVED = "identity_resolved"        # NEW
    FACE_AUTO_REGISTERED = "face_auto_registered"  # NEW
    ALERT = "alert"
```
