# Implementation Plan: Smart Recognition Pipeline

**Branch**: `001-smart-recognition-pipeline` | **Date**: 2026-02-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-smart-recognition-pipeline/spec.md`

## Summary

Fix the critical bug where door-crossing events are lost when face
recognition hasn't resolved the identity in time. Add self-learning
capabilities: auto-registration of unknown faces, priority-based face
recognition scheduling (door-approaching persons first), and auto-
verified training data collection for box detection improvement. All
changes stay within existing dependencies and must maintain 15 FPS AI
/ 30 FPS display on Jetson Orin Nano.

## Technical Context

**Language/Version**: Python 3.10+ (Jetson JetPack SDK)
**Primary Dependencies**: OpenCV, Ultralytics YOLO, InsightFace, NumPy, requests (no new deps)
**Storage**: Local filesystem (JSONL, CSV, NPZ for embeddings, JPEG for face/training images)
**Testing**: Manual integration testing on Jetson hardware; log-based verification
**Target Platform**: NVIDIA Jetson Orin Nano (aarch64 + CUDA, 8 GB VRAM); desktop x86_64 for development
**Project Type**: Single Python package (`inventory_monitor/`)
**Performance Goals**: >= 15 FPS AI processing, >= 30 FPS display, < 10s identity resolution delay
**Constraints**: 8 GB VRAM, offline-capable, no new dependencies, non-blocking retraining
**Scale/Scope**: Single camera feed, 1-5 concurrent persons in frame, ~100 events/day

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| # | Principle | Status | Notes |
|---|-----------|--------|-------|
| I | Self-Learning & Recognition | PASS | Core feature: auto-registration, incremental face learning, training data collection |
| II | Real-Time Performance | PASS | Priority scheduling adds negligible sort overhead; retraining is non-blocking; queue semantics unchanged |
| III | Edge-First Design | PASS | No new dependencies; no cloud requirement; all processing local; auto-register works offline |
| IV | Detection Reliability | PASS | Deferred events ensure zero missed events; priority scheduling improves identity resolution rate; confidence-lock mechanism unchanged |
| V | Privacy & Security | PASS | Face images stay local; identity-update events contain only labels (not embeddings); auto-registration metadata stored locally |
| VI | Simplicity | PASS | Extends existing classes (EventManager, FaceRecognizer, TrainingDataCollector); adds config fields to existing dataclasses; no new abstractions beyond DeferredEvent dict |

**Post-Phase 1 re-check**: All gates still pass. No new violations
introduced by the data model or contracts design.

## Project Structure

### Documentation (this feature)

```text
specs/001-smart-recognition-pipeline/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── api-events.md    # API event contracts
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
inventory_monitor/
├── __init__.py
├── __main__.py
├── app.py                  # MODIFY: deferred events, auto-register trigger, priority pass-through
├── config.py               # MODIFY: add RecognitionConfig fields, extend TrainingConfig
├── core/
│   ├── __init__.py
│   ├── event_manager.py    # MODIFY: add IDENTITY_RESOLVED + FACE_AUTO_REGISTERED event types,
│   │                       #          add deferred event tracking, add record_identity_resolved()
│   └── state_machine.py    # NO CHANGE (already fires events on position, not identity)
├── detectors/
│   ├── __init__.py
│   ├── box.py              # MODIFY: add auto-verification to TrainingDataCollector,
│   │                       #          add retrain trigger entry point
│   ├── face.py             # MODIFY: add auto_register(), priority-ordered recognition,
│   │                       #          add auto-registration metadata logging
│   └── person.py           # NO CHANGE
├── gui.py                  # MODIFY: add 'M' key for retraining trigger, update overlay
│                           #          for auto-registration notifications
├── trackers/
│   ├── __init__.py
│   └── bytetrack.py        # NO CHANGE
└── utils/
    ├── __init__.py
    ├── api_client.py        # NO CHANGE (already sends any event_type)
    ├── jetson.py            # NO CHANGE
    └── video.py             # NO CHANGE
```

**Structure Decision**: Single existing Python package. All changes are
modifications to existing files. No new files needed — the feature
extends existing classes and adds config fields.

## Complexity Tracking

> No constitution violations. No complexity justification needed.

## Key Implementation Details

### 1. Deferred Event Flow (FR-001, FR-002, FR-003)

```
Person crosses door → state_machine returns "entered"/"exited"
    → event_manager.record_entry/exit(identity=track.identity)
        → If identity is None:
            → Store DeferredEvent{track_id, event_type, timestamp}
            → Event still sent to API with identity=null

Later, face recognition resolves identity for track_id:
    → app._process_ai_result checks deferred events for this track
    → If deferred event exists and not resolved:
        → event_manager.record_identity_resolved(track_id, name, original_*)
        → Mark deferred event as resolved
        → API callback sends identity_resolved event
```

### 2. Priority Face Recognition (FR-006)

```
In AIWorker._process_frame:
    1. Compute priority score for each track:
       score = (1/distance_to_door) * (2 if approaching else 1)
    2. Sort tracks by score descending
    3. Process face recognition in order
    4. If time budget exceeded, skip remaining low-priority tracks
```

### 3. Auto-Registration (FR-004, FR-005, FR-007)

```
In app._process_ai_result, after face recognition:
    For each unrecognized track:
        if face_recognizer.get_registration_ready(track_id):
            if config.auto_register:
                label = config.auto_register_pattern.format(counter)
                face_recognizer.register_face(track_id, label)
                Log AutoRegistration metadata to JSONL
                Send FACE_AUTO_REGISTERED event
                counter += 1
            else:
                Show "Ready to register" notification on GUI
```

### 4. Auto-Verified Training Data (FR-008)

```
In TrainingDataCollector._process_sample:
    If all box detections have:
        confidence >= config.auto_verify_threshold AND
        detection_method == "yolo_custom"
    Then:
        Set verified=True, verification_method="auto_confidence"
```

### 5. Box Model Retraining (FR-009, FR-010)

```
GUI key 'M' pressed:
    1. Count verified samples in manifest
    2. If count >= config.retrain_min_samples:
        Launch subprocess: python -m ultralytics train ...
        Monitor in background thread
        On completion: reload model in box_detector
    3. Else: log "Not enough verified samples"
```
