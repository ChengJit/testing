# Implementation Plan: Full-Body Person Re-Identification Pipeline

**Branch**: `002-body-reid-pipeline` | **Date**: 2026-02-10 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-body-reid-pipeline/spec.md`

## Summary

Add full-body person re-identification with decoupled capture/recognition
threading. When a person crosses the door, the system captures their full
body (not just face), logs the event immediately, and asynchronously
resolves identity using face-first with body fallback. Body matching uses
hybrid OSNet embeddings + color histograms to identify people even when
facing away. Reworks run_monitor.py to share the same pipeline for both
--gui and --headless modes.

## Technical Context

**Language/Version**: Python 3.10+ (Jetson JetPack SDK)
**Primary Dependencies**: OpenCV, Ultralytics YOLO, InsightFace, NumPy, PyTorch (no new deps)
**Storage**: Local filesystem (NPZ for embeddings, JSON for profiles, JPEG for crops)
**Testing**: Manual integration testing on Jetson hardware; log-based verification
**Target Platform**: NVIDIA Jetson Orin Nano (aarch64 + CUDA, 8 GB VRAM); desktop x86_64 for dev
**Project Type**: Single Python package (`inventory_monitor/`)
**Performance Goals**: >= 15 FPS AI processing, >= 30 FPS display, < 10s identity resolution
**Constraints**: 8 GB VRAM, offline-capable, no new dependencies, non-blocking recognition
**Scale/Scope**: Single camera feed, 1-5 concurrent persons, ~100 events/day, ~20 known persons

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Principle | Status | Notes |
|---|-----------|--------|-------|
| I | Self-Learning & Recognition | PASS | Extends recognition with body embeddings; auto-registration preserved |
| II | Real-Time Performance | PASS | Dual-thread design ensures capture never blocks on recognition; queue-based decoupling |
| III | Edge-First Design | PASS | OSNet is 2.2MB; TensorRT optimized; no new dependencies; works offline |
| IV | Detection Reliability | PASS | Multi-modal matching (face → body → closest) increases identification rate |
| V | Privacy & Security | PASS | Body embeddings stored locally like face embeddings; no external transmission |
| VI | Simplicity | PASS | Reuses existing patterns (NPZ storage, embedding matching); minimal new abstractions |

**Post-Phase 1 re-check**: All gates still pass. Dual-thread model adds complexity
but is justified by the hard requirement that events log within 1 second while
recognition may take up to 10 seconds.

## Project Structure

### Documentation (this feature)

```text
specs/002-body-reid-pipeline/
├── plan.md              # This file
├── research.md          # Phase 0 output (body ReID approach, OSNet selection)
├── data-model.md        # Phase 1 output (PersonProfile, CaptureEvent, etc.)
├── quickstart.md        # Phase 1 output (usage guide)
├── contracts/
│   └── api-events.md    # API event schemas with match_method field
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
inventory_monitor/
├── __init__.py
├── __main__.py
├── app.py                  # MODIFY: integrate capture/recognition workers
├── config.py               # MODIFY: add BodyReIDConfig fields
├── core/
│   ├── __init__.py         # MODIFY: export new classes
│   ├── event_manager.py    # MODIFY: add match_method field, body scores
│   ├── state_machine.py    # NO CHANGE
│   ├── capture_worker.py   # NEW: door crossing detection, body crop queue
│   └── recognition_worker.py # NEW: async face + body recognition
├── detectors/
│   ├── __init__.py         # MODIFY: export BodyRecognizer
│   ├── box.py              # NO CHANGE
│   ├── face.py             # MINOR: add get_embedding_for_track()
│   ├── person.py           # NO CHANGE
│   └── body.py             # NEW: OSNet + histogram body ReID
├── gui.py                  # MODIFY: consume shared pipeline, display-only
├── trackers/
│   ├── __init__.py
│   └── bytetrack.py        # NO CHANGE
└── utils/
    ├── __init__.py
    ├── api_client.py       # MODIFY: include match_method in events
    ├── jetson.py           # NO CHANGE
    └── video.py            # NO CHANGE

run_monitor.py              # REWORK: unified entry point for --gui/--headless
```

**Structure Decision**: Single existing Python package. New files for workers
and body detector; modifications to existing files for integration. No new
top-level directories.

## Complexity Tracking

| Component | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Dual-thread (capture + recognition) | Spec requires <1s event logging while recognition takes up to 10s | Single-thread would either block capture or miss events |
| Queue between threads | Decouples capture from recognition throughput | Direct calls would couple their performance |
| Hybrid OSNet + histogram | Need 70%+ body accuracy; histogram alone is ~40-50% | Single method insufficient for target accuracy |

## Key Implementation Details

### 1. Body Re-Identification Approach (FR-004, FR-005)

```
Body Crop (from YOLO bbox)
    ├── Extract color histogram (HSV, 1152D)
    │       └── Chi-squared distance matching
    └── Extract OSNet embedding (128D, TensorRT)
            └── Cosine similarity matching

Combined score = 0.5 * histogram_score + 0.5 * osnet_score
```

### 2. Dual-Thread Architecture (FR-002, FR-003, FR-012)

```
Main Thread:
    video.read() → frame_buffer

Capture Worker Thread:
    while running:
        frame = frame_buffer.get()
        tracks = tracker.update(persons)
        for track crossing door:
            body_crop = extract_body_crop(frame, track)
            face_crop = extract_face_crop(frame, track)
            event_manager.record_entry/exit(identity=None)  # Immediate!
            capture_queue.put(CaptureEvent(...))

Recognition Worker Thread:
    while running:
        event = capture_queue.get()
        result = recognize(event)  # Face first, then body
        if result.identity:
            event_manager.record_identity_resolved(...)
            api_client.send_event(...)
```

### 3. Multi-Modal Recognition Flow (FR-004, FR-006, FR-007)

```
def recognize(capture_event):
    # 1. Try face first (if detected)
    if capture_event.face_crop is not None:
        face_result = face_recognizer.recognize(capture_event.face_crop)
        if face_result.confidence > 0.45:
            return RecognitionResult(
                identity=face_result.name,
                confidence=face_result.confidence,
                match_method="face"
            )

    # 2. Fall back to body matching
    body_result = body_recognizer.recognize(capture_event.body_crop)
    if body_result.confidence > 0.6:
        return RecognitionResult(
            identity=body_result.name,
            confidence=body_result.confidence,
            match_method="body"
        )

    # 3. Find closest match
    closest = body_recognizer.find_closest(capture_event.body_crop)
    if closest.confidence > 0.5:
        return RecognitionResult(
            identity=closest.name,
            confidence=closest.confidence,
            match_method="closest",
            needs_review=(closest.margin < 0.1)
        )

    # 4. Unknown - queue for auto-registration
    return RecognitionResult(identity=None, match_method="none")
```

### 4. Front/Back View Linking (FR-006)

```
Within same session (track_id lifecycle):
    - Entry with face visible → store as front view
    - Exit with back visible → link to same track

Cross-session:
    - On entry, if face matches known person:
        → Update their body_embedding with current view
    - On exit with back only:
        → Match body against known persons
        → If match, update their back_view embedding
```

### 5. Unified Run Modes (FR-010, FR-011)

```python
# run_monitor.py
def main():
    config = Config.load(args.config)

    # Shared pipeline (same for both modes)
    pipeline = MonitorPipeline(config)

    if args.headless:
        # No display, just run pipeline
        pipeline.run()
    else:
        # Wrap with GUI for display
        gui = MonitorGUI(pipeline)
        gui.run()
```

### 6. Body Embedding Storage (FR-005)

```python
# Parallel to face_embeddings.npz
body_embeddings.npz:
    names: ['Person_1', 'Person_2', ...]
    embeddings: [[128D], [128D], ...]  # OSNet
    histograms: [[1152D], [1152D], ...] # HSV

# Combined profile metadata
person_profiles.json:
    {
        "Person_1": {
            "has_face": true,
            "has_body_front": true,
            "has_body_back": true,
            "sample_count": 5
        }
    }
```

## Phase Dependencies

```
Phase 1: Config + Data Structures
    └── Phase 2: Body Recognizer
            └── Phase 3: Recognition Worker
                    └── Phase 4: Capture Worker
                            └── Phase 5: Pipeline Integration
                                    └── Phase 6: Run Mode Unification
                                            └── Phase 7: Testing + Polish
```

## Resource Impact

| Resource | Before | After | Delta |
|----------|--------|-------|-------|
| VRAM | ~700MB | ~780MB | +80MB |
| Models | YOLO, InsightFace | + OSNet (2.2MB) | +2.2MB |
| Latency | 25-45ms/frame | 28-50ms/frame | +3-5ms |
| FPS (AI) | 15-20 | 14-18 | -1-2 |
| Threads | 2 (main + AI) | 4 (main + AI + capture + recognition) | +2 |
