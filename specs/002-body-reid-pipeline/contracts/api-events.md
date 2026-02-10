# API Event Contracts: Body Re-Identification Pipeline

**Feature**: 002-body-reid-pipeline
**Date**: 2026-02-10
**Base URL**: `https://ops-portal.fasspay.com/report/cctv`

## Event Types

### Existing Events (unchanged)

| Event Type | Trigger | Description |
|------------|---------|-------------|
| `entered` | Door crossing inward | Person entered the room |
| `exited` | Door crossing outward | Person exited the room |
| `identity_resolved` | Deferred identity match | Identity determined after initial log |

### New/Enhanced Fields

All events now include optional body recognition data:

```json
{
  "timestamp": "2026-02-10T10:30:00.123Z",
  "event_type": "entered",
  "track_id": 42,
  "identity": "John Smith",
  "identity_confidence": 0.85,
  "match_method": "face",
  "box_count": 2,
  "direction": "entering",
  "camera_id": "warehouse-cam-001",
  "details": {
    "face_score": 0.85,
    "body_score": 0.72,
    "needs_review": false
  }
}
```

## Event Schemas

### Entry/Exit Event

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["timestamp", "event_type", "track_id", "camera_id"],
  "properties": {
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "event_type": {
      "type": "string",
      "enum": ["entered", "exited"]
    },
    "track_id": {
      "type": "integer",
      "minimum": 0
    },
    "identity": {
      "type": ["string", "null"],
      "description": "Null if not yet resolved"
    },
    "identity_confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "match_method": {
      "type": "string",
      "enum": ["face", "body", "closest", "none"],
      "description": "How identity was determined"
    },
    "box_count": {
      "type": "integer",
      "minimum": 0
    },
    "direction": {
      "type": "string",
      "enum": ["entering", "exiting"]
    },
    "camera_id": {
      "type": "string"
    },
    "image_base64": {
      "type": ["string", "null"],
      "description": "Optional JPEG frame"
    },
    "details": {
      "type": "object",
      "properties": {
        "face_score": { "type": "number" },
        "body_score": { "type": "number" },
        "needs_review": { "type": "boolean" },
        "entry_box_count": { "type": "integer" },
        "exit_box_count": { "type": "integer" },
        "box_difference": { "type": "integer" }
      }
    }
  }
}
```

### Identity Resolved Event

Sent when deferred identity is determined after initial event logging.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["timestamp", "event_type", "track_id", "identity", "camera_id"],
  "properties": {
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "event_type": {
      "const": "identity_resolved"
    },
    "track_id": {
      "type": "integer"
    },
    "identity": {
      "type": "string",
      "description": "Resolved identity name"
    },
    "identity_confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "match_method": {
      "type": "string",
      "enum": ["face", "body", "closest"]
    },
    "camera_id": {
      "type": "string"
    },
    "details": {
      "type": "object",
      "properties": {
        "original_event_type": {
          "type": "string",
          "enum": ["entered", "exited"]
        },
        "original_timestamp": {
          "type": "string",
          "format": "date-time"
        },
        "resolution_delay_ms": {
          "type": "integer",
          "description": "Time from original event to resolution"
        },
        "face_score": { "type": "number" },
        "body_score": { "type": "number" }
      }
    }
  }
}
```

## API Endpoints

### POST /event

Send single event (existing behavior, enhanced payload).

**Request**:
```http
POST /report/cctv/event
Content-Type: application/json

{
  "timestamp": "2026-02-10T10:30:00.123Z",
  "event_type": "entered",
  "track_id": 42,
  "identity": null,
  "box_count": 1,
  "camera_id": "warehouse-cam-001",
  "match_method": "none"
}
```

**Response**:
```json
{
  "event_id": "evt_abc123",
  "status": "accepted"
}
```

### POST /events/batch

Send multiple events (existing behavior).

### POST /heartbeat

Unchanged from current implementation.

## Match Method Values

| Value | Description | Typical Confidence |
|-------|-------------|-------------------|
| `face` | Matched via face recognition | 0.75-1.0 |
| `body` | Matched via body appearance (OSNet + histogram) | 0.60-0.85 |
| `closest` | Best match below normal threshold | 0.50-0.70 |
| `none` | No match found, identity unknown | N/A |

## Backward Compatibility

- All new fields are optional
- Existing API consumers can ignore new fields
- `match_method` defaults to "face" if not specified
- `details.face_score` and `details.body_score` are informational only
