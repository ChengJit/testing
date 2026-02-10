# Tasks: Full-Body Person Re-Identification Pipeline

**Input**: Design documents from `/specs/002-body-reid-pipeline/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `inventory_monitor/` at repository root
- Paths use forward slashes for cross-platform compatibility

---

## Phase 1: Setup (Configuration & Data Structures)

**Purpose**: Add configuration fields and data structures needed by all user stories

- [x] T001 Add BodyReIDConfig fields to inventory_monitor/config.py (body_reid_enabled, body_model, body_threshold, use_color_histogram, face_body_weight)
- [x] T002 Add body_embeddings_file and person_profiles_file paths to Config.save/load in inventory_monitor/config.py
- [x] T003 [P] Create CaptureEvent dataclass in inventory_monitor/core/event_manager.py (track_id, timestamp, event_type, full_body_crop, face_crop, face_detected, box_count, processing_status)
- [x] T004 [P] Create RecognitionResult dataclass in inventory_monitor/core/event_manager.py (track_id, identity, confidence, match_method, face_score, body_score, needs_review)
- [x] T005 Update core/__init__.py to export CaptureEvent and RecognitionResult

---

## Phase 2: Foundational (Body Recognition Module)

**Purpose**: Create the body recognition detector that all user stories depend on

**⚠️ CRITICAL**: User story work cannot begin until body recognition is functional

- [x] T006 Create inventory_monitor/detectors/body.py with BodyRecognizer class skeleton (init, recognize, find_closest methods)
- [x] T007 Implement extract_color_histogram() in inventory_monitor/detectors/body.py (HSV 18x8x8 bins, 1152D vector, cv2.calcHist)
- [x] T008 Implement histogram_match() in inventory_monitor/detectors/body.py (Chi-squared distance, normalized 0-1 score)
- [x] T009 [P] Implement load_osnet_model() in inventory_monitor/detectors/body.py (PyTorch model loading, TensorRT optional)
- [x] T010 Implement extract_body_embedding() in inventory_monitor/detectors/body.py (128D OSNet embedding, L2 normalization)
- [x] T011 Implement combined_score() in inventory_monitor/detectors/body.py (0.5 * histogram + 0.5 * osnet)
- [x] T012 Implement load_embeddings() and save_embeddings() in inventory_monitor/detectors/body.py (NPZ format matching face.py pattern)
- [x] T013 Implement recognize() method in inventory_monitor/detectors/body.py (compare query against all known, return best match)
- [x] T014 Update detectors/__init__.py to export BodyRecognizer

**Checkpoint**: BodyRecognizer module complete - user stories can now proceed

---

## Phase 3: User Story 1 - Identify People Facing Away (Priority: P1) 🎯 MVP

**Goal**: Recognize people by body appearance when face is not visible

**Independent Test**: Have a known person walk through door facing away; system should identify them by body matching

### Implementation for User Story 1

- [x] T015 [US1] Implement find_closest() in inventory_monitor/detectors/body.py (top-K with margin check, needs_review flag for <10% margin)
- [x] T016 [US1] Add register_body() method to BodyRecognizer in inventory_monitor/detectors/body.py (store embedding + histogram for new person)
- [x] T017 [US1] Add update_body_embedding() to BodyRecognizer in inventory_monitor/detectors/body.py (average new sample with existing)
- [x] T018 [US1] Implement body crop extraction helper in inventory_monitor/app.py (extract_body_crop from YOLO bbox)
- [x] T019 [US1] Add body recognition call in AIWorker._process_frame() in inventory_monitor/app.py (after face recognition fails)
- [x] T020 [US1] Add body_score and match_method fields to event details in inventory_monitor/core/event_manager.py
- [x] T021 [US1] Add match_method to API event payload in inventory_monitor/utils/api_client.py (face/body/closest/none)

**Checkpoint**: Body-based recognition working for people facing away

---

## Phase 4: User Story 2 - Immediate Event Logging with Deferred Identity (Priority: P1)

**Goal**: Log events within 1 second, resolve identity asynchronously in background thread

**Independent Test**: Person walks quickly through door; event logged <1s, identity resolved <10s

### Implementation for User Story 2

- [x] T022 [P] [US2] Create RecognitionQueue class in inventory_monitor/core/recognition_worker.py (thread-safe Queue, max_size=20, drop_oldest)
- [x] T023 [US2] Create RecognitionWorker class skeleton in inventory_monitor/core/recognition_worker.py (init, start, stop, _process_loop)
- [x] T024 [US2] Implement _process_event() in RecognitionWorker in inventory_monitor/core/recognition_worker.py (face first → body → closest → unknown)
- [x] T025 [US2] Implement _send_identity_resolved() in RecognitionWorker in inventory_monitor/core/recognition_worker.py (call event_manager.record_identity_resolved)
- [x] T026 [P] [US2] Create CaptureWorker class in inventory_monitor/core/capture_worker.py (monitors door crossings, extracts crops, queues events)
- [x] T027 [US2] Implement _on_door_crossing() in CaptureWorker in inventory_monitor/core/capture_worker.py (immediate log with identity=None, queue CaptureEvent)
- [x] T028 [US2] Integrate CaptureWorker and RecognitionWorker into InventoryMonitor in inventory_monitor/app.py
- [x] T029 [US2] Update core/__init__.py to export CaptureWorker and RecognitionWorker

**Checkpoint**: Dual-thread architecture working - events logged immediately, identity resolved async

---

## Phase 5: User Story 3 - Multi-Modal Recognition with Fallback (Priority: P2)

**Goal**: Try face → body → closest match in order of reliability

**Independent Test**: Test with face visible, partially occluded, and hidden; all scenarios produce identification

### Implementation for User Story 3

- [ ] T030 [US3] Add face_recognizer.get_embedding_for_crop() helper in inventory_monitor/detectors/face.py (direct embedding extraction from crop)
- [ ] T031 [US3] Implement multi-modal recognition flow in RecognitionWorker._process_event() in inventory_monitor/core/recognition_worker.py
- [ ] T032 [US3] Add fallback logic: if face_score < 0.45, try body in inventory_monitor/core/recognition_worker.py
- [ ] T033 [US3] Add closest match fallback: if body_score < 0.6, try find_closest in inventory_monitor/core/recognition_worker.py
- [ ] T034 [US3] Add auto-registration trigger for truly unknown persons in inventory_monitor/core/recognition_worker.py (collect samples → register)
- [ ] T035 [US3] Update RecognitionResult to include alternative_matches in inventory_monitor/core/event_manager.py

**Checkpoint**: Multi-modal recognition with intelligent fallback working

---

## Phase 6: User Story 4 - Unified Pipeline for GUI and Headless Modes (Priority: P2)

**Goal**: Same capture/recognition pipeline works in both --gui and --headless modes

**Independent Test**: Run same scenario in both modes, compare logs - should be identical

### Implementation for User Story 4

- [ ] T036 [US4] Create MonitorPipeline class in inventory_monitor/app.py (shared pipeline logic, no display code)
- [ ] T037 [US4] Refactor InventoryMonitor to use MonitorPipeline in inventory_monitor/app.py
- [ ] T038 [US4] Update InventoryGUI to wrap MonitorPipeline in inventory_monitor/gui.py (display-only, no pipeline logic)
- [ ] T039 [US4] Rework run_monitor.py to use unified entry point (MonitorPipeline for headless, InventoryGUI wrapping for GUI)
- [ ] T040 [US4] Remove duplicate pipeline code from InventoryGUI in inventory_monitor/gui.py
- [ ] T041 [US4] Add --mode argument to run_monitor.py as alternative to --gui/--headless

**Checkpoint**: GUI and headless modes share identical pipeline logic

---

## Phase 7: User Story 5 - Front/Back View Cross-Referencing (Priority: P3)

**Goal**: Link front view (face visible) with back view (body only) for same person

**Independent Test**: New person enters facing camera, exits with back; system links both views

### Implementation for User Story 5

- [ ] T042 [US5] Add view_type field to BodyEmbedding storage in inventory_monitor/detectors/body.py (front/back/side)
- [ ] T043 [US5] Implement store_front_view() in BodyRecognizer in inventory_monitor/detectors/body.py (save with view_type="front")
- [ ] T044 [US5] Implement store_back_view() in BodyRecognizer in inventory_monitor/detectors/body.py (save with view_type="back")
- [ ] T045 [US5] Add front/back linking logic in RecognitionWorker in inventory_monitor/core/recognition_worker.py (entry+face → store front, exit+body_only → link back)
- [ ] T046 [US5] Update person_profiles.json schema to track has_body_front, has_body_back flags in inventory_monitor/detectors/body.py
- [ ] T047 [US5] Add get_person_views() method in BodyRecognizer in inventory_monitor/detectors/body.py (return both front and back crops)

**Checkpoint**: Front/back views linked for complete person profiles

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T048 [P] Update config.json.example with new body ReID fields
- [ ] T049 [P] Add logging for body recognition events (DEBUG level) in inventory_monitor/detectors/body.py
- [ ] T050 Add error handling for missing OSNet model (graceful fallback to histogram-only) in inventory_monitor/detectors/body.py
- [ ] T051 [P] Add queue size monitoring and warning logs in inventory_monitor/core/recognition_worker.py
- [ ] T052 Verify FPS impact stays within 15+ target on Jetson hardware
- [ ] T053 Update Config.save() to persist new body ReID fields in inventory_monitor/config.py
- [ ] T054 Run quickstart.md validation - verify all documented features work

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - can start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 - BLOCKS all user stories
- **Phase 3-7 (User Stories)**: All depend on Phase 2 completion
  - US1 and US2 can proceed in parallel (both P1)
  - US3 and US4 can proceed in parallel after US1/US2 (both P2)
  - US5 depends on US1 body recognition (P3)
- **Phase 8 (Polish)**: Depends on all user stories being complete

### User Story Dependencies

| Story | Priority | Dependencies | Can Parallel With |
|-------|----------|--------------|-------------------|
| US1 (Body Recognition) | P1 | Phase 2 only | US2 |
| US2 (Deferred Identity) | P1 | Phase 2 only | US1 |
| US3 (Multi-Modal) | P2 | US1 (body recognition) | US4 |
| US4 (Unified Pipeline) | P2 | US1, US2 (workers exist) | US3 |
| US5 (Front/Back Views) | P3 | US1 (body storage) | - |

### Within Each User Story

- Core functionality before integration
- Storage/data before logic
- Workers before pipeline integration

### Parallel Opportunities

**Phase 1 (Setup)**:
```
T003, T004 can run in parallel (different dataclasses)
```

**Phase 2 (Foundational)**:
```
T007, T009 can run in parallel (histogram vs OSNet)
```

**User Stories (after Phase 2)**:
```
US1 and US2 can proceed simultaneously
US3 and US4 can proceed simultaneously (after US1/US2)
```

---

## Parallel Example: User Story 2

```bash
# Launch foundational workers in parallel:
Task: "Create RecognitionQueue class in inventory_monitor/core/recognition_worker.py"
Task: "Create CaptureWorker class in inventory_monitor/core/capture_worker.py"

# Then integrate sequentially:
Task: "Integrate CaptureWorker and RecognitionWorker into InventoryMonitor"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2)

1. Complete Phase 1: Setup (config + dataclasses)
2. Complete Phase 2: Foundational (BodyRecognizer module)
3. Complete Phase 3: US1 - Body recognition working
4. Complete Phase 4: US2 - Dual-thread pipeline working
5. **STOP and VALIDATE**: Test body recognition + deferred identity
6. Deploy/demo if ready - this is a functional MVP

### Incremental Delivery

1. Setup + Foundational → Core infrastructure ready
2. Add US1 + US2 → MVP (body recognition + deferred logging)
3. Add US3 → Multi-modal recognition with fallbacks
4. Add US4 → Unified GUI/headless modes
5. Add US5 → Complete person profiles with front/back views
6. Polish → Production-ready

### Parallel Team Strategy

With 2 developers after Phase 2:
- Developer A: US1 (body recognition) → US3 (multi-modal)
- Developer B: US2 (threading) → US4 (unified pipeline)
- Both: US5 (depends on US1), Phase 8 (polish)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- Each user story is independently testable after completion
- Commit after each task or logical group
- Stop at any checkpoint to validate independently
- OSNet model is optional - system falls back to histogram-only if unavailable
