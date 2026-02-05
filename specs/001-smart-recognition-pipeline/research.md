# Research: Smart Recognition Pipeline

**Branch**: `001-smart-recognition-pipeline` | **Date**: 2026-02-05

## R1: Deferred Event Architecture

**Decision**: Implement a deferred event enrichment pattern where events
are logged immediately with whatever identity is available, and a
follow-up `IDENTITY_RESOLVED` event is sent when recognition completes.

**Rationale**: The current system fires `record_entry`/`record_exit` in
`_process_ai_result` only when the state machine transitions. If the
face is unrecognized at that moment, the identity field is `None`. The
event still gets logged locally (CSV/JSON) and sent to the API, but
consumers see `identity=null`. The fix has two parts:

1. **Always fire the event**: The current code already does this — the
   state machine transitions on position, not identity. However,
   `_process_ai_result` in `app.py:616-657` only records events when
   `event == "entered"` or `event == "exited"`. The identity passed is
   `track.identity`, which may still be `None`. This is correct behavior
   — we just need to ensure the API callback handles `identity=None`
   gracefully (it already does).

2. **Send identity-update**: Add a new `EventType.IDENTITY_RESOLVED`
   event type. When face recognition resolves an identity for a track
   that has already crossed the door (i.e., has a logged entry/exit with
   `identity=None`), fire an `IDENTITY_RESOLVED` event with the
   track_id and resolved name. The API client sends this as a separate
   event.

3. **Local log enrichment**: Optionally update the CSV/JSON records
   retroactively. Since JSONL is append-only, this means appending a
   correction record rather than rewriting. CSV rewrite is impractical
   for streaming; instead, the identity-update event serves as the
   correction record.

**Alternatives considered**:
- **Delay event until identity resolves**: Rejected. Adds unbounded
  latency and risks losing the event entirely if the face is never
  recognized.
- **Buffer events with timeout**: Rejected. Adds complexity (timeout
  logic, buffer management) without solving the case where identity
  never resolves. Violates Simplicity principle.

---

## R2: Face Recognition Priority Scheduling

**Decision**: Implement distance-to-door priority scoring in the AI
worker so that tracks closer to the door line receive face recognition
first.

**Rationale**: Currently `AIWorker._process_frame` iterates all tracks
in arbitrary `dict` order for face recognition (app.py:209-222). On
Jetson with limited GPU, processing all faces takes time. If the person
approaching the door is processed last, their identity may not resolve
before the door crossing event.

**Approach**: Sort tracks by a priority score before face recognition:
- `priority = 1.0 / max(1, abs(track.center[1] - door_y))`
- Tracks approaching the door (based on velocity direction) get a 2x
  multiplier.
- Process in descending priority order.
- If time budget is exhausted (process_interval), skip remaining
  low-priority tracks.

This requires passing `door_y` and `enter_direction_down` to the AI
worker (or a priority calculator function).

**Alternatives considered**:
- **Dedicated face recognition thread**: Rejected. Adds thread
  coordination complexity and doesn't solve ordering; just adds
  parallelism that may contend on GPU.
- **Pre-crop and batch faces**: Could help throughput but doesn't
  address ordering. May be a future optimization.

---

## R3: Auto-Registration Strategy

**Decision**: Extend the existing `FaceRecognizer` with an
auto-registration mode that activates when
`samples_for_registration` samples are collected and a configurable
`auto_register` flag is enabled.

**Rationale**: The existing code in `face.py` already collects unknown
face samples in `_collect_unknown_face` and checks readiness via
`get_registration_ready`. The missing piece is triggering registration
automatically instead of waiting for the operator to press 'R' in the
GUI.

**Approach**:
1. Add `auto_register: bool = False` and
   `auto_register_pattern: str = "Person_{:03d}"` to config.
2. In `_process_ai_result`, after face recognition, check if any
   unrecognized track has `get_registration_ready() == True`.
3. If `auto_register` is enabled, call `register_face(track_id, label)`
   with a generated label. Increment a persistent counter for unique
   label generation.
4. If `auto_register` is disabled, add a visual notification on the GUI
   overlay indicating "New face ready for registration".
5. Store auto-registration metadata (generated label, timestamp, reviewed
   flag) in a JSONL file for operator review.

**Alternatives considered**:
- **Cluster-based registration (group similar unknowns)**: Rejected for
  now. Adds complexity and requires comparing embeddings across tracks.
  The simple threshold-based approach handles the common single-new-
  person scenario.
- **Server-side registration**: Rejected. Violates Edge-First principle.

---

## R4: Box Model Retraining Strategy

**Decision**: Enhance `TrainingDataCollector` to auto-verify
high-confidence samples and add a retraining entry point callable from
the GUI.

**Rationale**: The existing `TrainingDataCollector` saves all samples
(positive and negative) but marks them as `verified: False` in the
manifest. The missing pieces are:

1. **Auto-verification**: Samples detected by the custom YOLO model with
   confidence above a configurable threshold (e.g., 0.8) can be marked
   `verified: True` automatically.
2. **Retraining trigger**: Add a keyboard shortcut or config-driven
   trigger that runs YOLO training on verified samples. This should run
   in a background process (not thread, since YOLO training is
   GPU-intensive and long-running).
3. **Model hot-swap**: After retraining completes, the new model weights
   are saved alongside the old ones. The box detector reloads the new
   model on the next frame cycle.

**Alternatives considered**:
- **Continuous online learning**: Rejected. YOLO fine-tuning requires
  batch processing and is not suitable for frame-by-frame updates.
- **Cloud-based training**: Rejected. Violates Edge-First principle and
  requires connectivity.

---

## R5: API Identity-Update Event

**Decision**: Add a new event type `identity_resolved` to the existing
API event schema. The event includes `track_id`, `identity`,
`identity_confidence`, `original_event_type`, and
`original_timestamp`.

**Rationale**: The existing API endpoint (`/event` and `/events/batch`)
accepts any `event_type` string. Adding a new event type requires no
API schema changes on the server side — the ops-portal just needs to
handle the new event type in its event processing pipeline. If it
doesn't recognize it, it should store it as-is (standard practice for
event-driven systems).

**Alternatives considered**:
- **PATCH endpoint to update existing event**: Rejected. Requires the
  API to support event mutation, which is atypical for event logs and
  may not be implemented.
- **Embed identity updates in existing event details field**: Rejected.
  Loses the explicit event semantics and makes downstream processing
  harder.

---

## R6: New Dependencies Assessment

**Decision**: No new runtime dependencies required.

**Rationale**: All features can be implemented using existing
dependencies (OpenCV, NumPy, Ultralytics YOLO, InsightFace, requests).
Box model retraining uses the existing Ultralytics YOLO training API.
Auto-registration uses the existing InsightFace embedding pipeline.
Deferred events use the existing event manager and API client.

This aligns with Constitution Principle VI (Simplicity) — no new
dependencies.
