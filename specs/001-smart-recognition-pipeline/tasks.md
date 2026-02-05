# Tasks: Smart Recognition Pipeline

**Input**: Design documents from `/specs/001-smart-recognition-pipeline/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Not explicitly requested in the feature specification. Test tasks are omitted.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `inventory_monitor/` at repository root (`cctv/`)
- All file paths are relative to `cctv/inventory_monitor/` unless otherwise noted

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add new config fields and event types that multiple user stories depend on

- [ ] T001 Add new config fields to DetectionConfig and TrainingConfig dataclasses in inventory_monitor/config.py: add `auto_register` (bool, default False), `auto_register_pattern` (str, default "Person_{}"), `priority_recognition` (bool, default True), `identity_resolve_timeout` (float, default 30.0) to DetectionConfig; add `auto_verify_threshold` (float, default 0.8), `retrain_min_samples` (int, default 500) to TrainingConfig. Update Config.load() and Config.save() to handle these new fields.
- [ ] T002 Add IDENTITY_RESOLVED and FACE_AUTO_REGISTERED enum members to EventType in inventory_monitor/core/event_manager.py: `IDENTITY_RESOLVED = "identity_resolved"` and `FACE_AUTO_REGISTERED = "face_auto_registered"`
- [ ] T003 Export new EventType members from inventory_monitor/core/__init__.py if not already using wildcard export

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core deferred event infrastructure that US1 depends on and US2/US4 will also use

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Add DeferredEvent dataclass to inventory_monitor/core/event_manager.py with fields: track_id (int), event_type (EventType), timestamp (str), box_count (int), identity (Optional[str] = None), identity_confidence (float = 0.0), resolved (bool = False), resolved_timestamp (Optional[str] = None)
- [ ] T005 Add deferred event tracking dict (`_deferred_events: Dict[int, DeferredEvent]`) and timeout-based cleanup method (`cleanup_deferred_events(timeout: float)`) to EventManager in inventory_monitor/core/event_manager.py
- [ ] T006 Add `record_identity_resolved()` method to EventManager in inventory_monitor/core/event_manager.py: accepts track_id, identity, identity_confidence, original_event_type, original_timestamp; creates IDENTITY_RESOLVED event with details dict containing original_event_type, original_timestamp, resolution_delay_ms; marks the deferred event as resolved
- [ ] T007 Update `record_entry()` and `record_exit()` methods in EventManager (inventory_monitor/core/event_manager.py) to create a DeferredEvent when the passed identity is None, storing it in _deferred_events keyed by track_id

**Checkpoint**: Foundation ready — deferred event infrastructure is in place. User story implementation can now begin.

---

## Phase 3: User Story 1 — Log All Entries and Exits (Priority: P1) MVP

**Goal**: Every door-crossing event is logged and sent to API immediately, even with unknown identity. Follow-up identity-update events are sent when recognition resolves.

**Independent Test**: Have a person walk through the door while face recognition is still processing. Check logs/events.jsonl for the immediate entry/exit event (identity=null) followed by an identity_resolved event once the face is matched.

### Implementation for User Story 1

- [ ] T008 [US1] Update `_process_ai_result()` in inventory_monitor/app.py to check for deferred events after face recognition: when a track receives an identity (result.faces[track_id]) AND that track_id has an unresolved deferred event in event_manager._deferred_events, call event_manager.record_identity_resolved() with the resolved identity and the original event metadata
- [ ] T009 [US1] Add periodic deferred event cleanup in inventory_monitor/app.py: in the main processing loop (process_one_frame or run), call event_manager.cleanup_deferred_events(config.detection.identity_resolve_timeout) every N frames (e.g., every 30 frames) to expire stale deferred events
- [ ] T010 [US1] Verify that the existing _api_event_callback in inventory_monitor/core/event_manager.py correctly handles the new IDENTITY_RESOLVED event type by sending it through the API client (it should already work since send_event accepts any event_type string, but verify the details dict is included)
- [ ] T011 [US1] Add logging for deferred event creation, resolution, and expiry in inventory_monitor/core/event_manager.py using the existing logger: log at INFO level when a deferred event is created (track_id, event_type), resolved (track_id, identity, delay_ms), or expired (track_id, age_seconds)

**Checkpoint**: At this point, 100% of door-crossing events are logged regardless of identity status. Identity-resolved follow-up events are sent. US1 is fully functional.

---

## Phase 4: User Story 4 — Reduce Face Recognition Latency (Priority: P2)

**Goal**: Face recognition prioritizes tracks approaching the door zone so identities resolve before door crossing.

**Independent Test**: With multiple people in frame, observe that the person walking toward the door gets identified before a stationary person inside. Measure time from first detection to identity display — target is resolution before door crossing 80% of the time.

### Implementation for User Story 4

- [ ] T012 [US4] Add door_y and enter_direction_down parameters to AIWorker.__init__() in inventory_monitor/app.py so the worker can compute recognition priority. Pass these from InventoryMonitor when creating AIWorker (requires deferring AIWorker creation until after _init_managers, or updating the values after first frame)
- [ ] T013 [US4] Add `_compute_priority(track, door_y, enter_direction_down) -> float` helper method to AIWorker in inventory_monitor/app.py: compute `base = 1.0 / max(1, abs(track.center[1] - door_y))`; determine if approaching based on velocity direction and enter_direction_down; return `base * 2.0 if approaching else base`
- [ ] T014 [US4] Update `_process_frame()` in AIWorker (inventory_monitor/app.py) to sort tracks by priority score (descending) before the face recognition loop (step 2). Replace `for track_id, track in tracks.items()` with sorted iteration. Only apply sorting when config.detection.priority_recognition is True.
- [ ] T015 [US4] Add optional time-budget check in the face recognition loop in AIWorker._process_frame() (inventory_monitor/app.py): if elapsed time since start of face recognition exceeds 80% of self.process_interval, skip remaining low-priority tracks and log which tracks were skipped

**Checkpoint**: Face recognition now processes door-approaching persons first. Combined with US1's deferred events, most entries have identity resolved before crossing.

---

## Phase 5: User Story 2 — Self-Learning Face Recognition (Priority: P2)

**Goal**: Unknown faces are auto-collected and optionally auto-registered with generated labels. Operators can correct labels via GUI.

**Independent Test**: Introduce an unregistered person to the camera. After 5 appearances, observe auto-registration with a "Person_NNN" label (if auto_register=true) or a "Ready to register" GUI notification (if auto_register=false). Correct the label via C key and verify the person is recognized on next appearance.

### Implementation for User Story 2

- [ ] T016 [US2] Add `auto_register()` method to FaceRecognizer in inventory_monitor/detectors/face.py: accepts track_id; generates a label using the auto_register_pattern and a persistent counter; calls self.register_face(track_id, label); returns the generated label. The counter is loaded from data/auto_registrations.jsonl on startup (count existing entries) and incremented on each call.
- [ ] T017 [US2] Add `_log_auto_registration()` method to FaceRecognizer in inventory_monitor/detectors/face.py: appends an AutoRegistration record (generated_label, track_id, sample_count, registration_time, reviewed=False) as a JSON line to data/auto_registrations.jsonl. Create the file and parent directory if they don't exist.
- [ ] T018 [US2] Add `_load_auto_register_counter()` method to FaceRecognizer.__init__() in inventory_monitor/detectors/face.py: read data/auto_registrations.jsonl on startup; count entries to determine next counter value; store as self._auto_register_counter (int)
- [ ] T019 [US2] Update `_process_ai_result()` in inventory_monitor/app.py to trigger auto-registration: after the face recognition loop, iterate unrecognized tracks; if face_recognizer.get_registration_ready(track_id) is True, check config.detection.auto_register; if True, call face_recognizer.auto_register(track_id), send FACE_AUTO_REGISTERED event via event_manager, and set track identity; if False, set a flag for GUI notification
- [ ] T020 [US2] Add auto-registration notification overlay in inventory_monitor/gui.py: when the app has unregistered-but-ready tracks (flag from T019), display "New face ready — press R to register" text near the stats panel in _draw_overlay()
- [ ] T021 [US2] Update the existing _apply_correction() flow in inventory_monitor/app.py to also update the auto_registrations.jsonl: when an operator corrects a label that was auto-generated, append a correction record (corrected_label, corrected_time) to the JSONL file and call face_recognizer.retrain_all() as already implemented
- [ ] T022 [US2] Add low-margin match flagging to FaceRecognizer.recognize() in inventory_monitor/detectors/face.py: when the best match score is above recognition_threshold but within 0.1 of the second-best match, log a WARNING and include a "low_margin" flag in the FaceMatch result for downstream handling

**Checkpoint**: Self-learning face recognition is active. New faces are auto-registered (or prompted). Operators can correct labels. Low-margin matches are flagged.

---

## Phase 6: User Story 3 — Self-Learning Box Detection (Priority: P3)

**Goal**: High-confidence box detections are auto-verified as training data. Retraining can be triggered from the GUI.

**Independent Test**: Enable training collection (training.enabled=true). Observe that manifest.jsonl entries from the custom YOLO model with confidence >= 0.8 have `"verified": true`. Press M in the GUI to trigger retraining when enough samples are collected.

### Implementation for User Story 3

- [ ] T023 [US3] Update `_process_sample()` in TrainingDataCollector (inventory_monitor/detectors/box.py) to auto-verify samples: after writing the manifest entry, check if all box_detections have confidence >= config auto_verify_threshold AND detection_method == "yolo_custom"; if so, set verified=True and verification_method="auto_confidence" in the manifest entry. Add the auto_verify_threshold parameter to TrainingDataCollector.__init__().
- [ ] T024 [US3] Add `count_verified_samples()` method to TrainingDataCollector in inventory_monitor/detectors/box.py: read manifest.jsonl and count entries where verified=True; return the count
- [ ] T025 [US3] Add `trigger_retrain(model_output_path: str)` method to TrainingDataCollector in inventory_monitor/detectors/box.py: verify count_verified_samples() >= retrain_min_samples; generate a dataset.yaml from verified manifest entries; launch `python -m ultralytics ... train` as a subprocess (subprocess.Popen); store the process handle; return True/False. Add retrain_min_samples parameter to __init__().
- [ ] T026 [US3] Add `check_retrain_complete()` method to TrainingDataCollector in inventory_monitor/detectors/box.py: poll the stored subprocess handle; if complete and exit code 0, return the path to the new model weights; otherwise return None
- [ ] T027 [US3] Add `reload_model(model_path: str)` method to BoxDetector in inventory_monitor/detectors/box.py: load the new YOLO model weights and replace self.custom_model; log the swap at INFO level
- [ ] T028 [US3] Add 'M' keyboard shortcut handler in inventory_monitor/app.py _handle_key(): if training_collector exists, call trigger_retrain(); log result. In the main loop, periodically call check_retrain_complete() and if a new model is ready, call box_detector.reload_model().
- [ ] T029 [US3] Add 'M' key to the instructions bar in inventory_monitor/gui.py _draw_overlay(): update the bottom bar text to include "M:Retrain" when training_collector is active

**Checkpoint**: Box detection self-learning is active. High-confidence samples are auto-verified. Retraining is triggerable from GUI and model hot-swaps on completion.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T030 Add the new config fields (auto_register, auto_register_pattern, priority_recognition, identity_resolve_timeout, auto_verify_threshold, retrain_min_samples) to Config.save() serialization in inventory_monitor/config.py so they persist to config.json
- [ ] T031 Run quickstart.md validation: manually walk through all verification steps in specs/001-smart-recognition-pipeline/quickstart.md to confirm each feature works end-to-end
- [ ] T032 Review all new logging statements across modified files for consistency: ensure INFO level for significant events (deferred creation, resolution, auto-registration, retraining); DEBUG level for per-frame operations; WARNING for low-margin matches and skipped tracks; verify no raw embeddings or face images are logged (Privacy principle)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 (T001-T003) — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Phase 2 (T004-T007) — No dependencies on other stories
- **US4 (Phase 4)**: Depends on Phase 1 (T001) — can run in parallel with US1
- **US2 (Phase 5)**: Depends on Phase 2 (T004-T007) — can run in parallel with US1/US4
- **US3 (Phase 6)**: Depends on Phase 1 (T001) — can run in parallel with other stories
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Depends on Foundational phase (deferred event infrastructure). No deps on other stories.
- **User Story 4 (P2)**: Depends only on config fields (Phase 1). Independent of US1's deferred events — improves recognition speed separately. Can run in parallel with US1.
- **User Story 2 (P2)**: Depends on Foundational phase (uses event_manager for FACE_AUTO_REGISTERED events). Independent of US1/US4.
- **User Story 3 (P3)**: Depends only on config fields (Phase 1). Fully independent of all other stories.

### Within Each User Story

- Config additions before feature code
- Core logic before integration points
- Data layer before API layer
- Feature implementation before GUI updates

### Parallel Opportunities

- T001, T002, T003 can all run in parallel (different files)
- T004, T005, T006, T007 are sequential (same file: event_manager.py)
- US1 (T008-T011) and US4 (T012-T015) can run in parallel (US4 touches app.py AIWorker; US1 touches app.py InventoryMonitor — different sections but same file, so sequential is safer)
- US2 (T016-T022) and US3 (T023-T029) can run in parallel (different files: face.py vs box.py)
- Within US2: T016, T017, T018 can run in parallel (all add methods to face.py, but sequential is safer for same-file edits)
- Within US3: T023-T026 are sequential (same file: box.py); T028 and T029 can run in parallel (app.py vs gui.py)
- T030, T031, T032 can all run in parallel

---

## Parallel Example: Phase 1 Setup

```bash
# All three setup tasks touch different files — run in parallel:
Task: "T001 - Add config fields in inventory_monitor/config.py"
Task: "T002 - Add EventType members in inventory_monitor/core/event_manager.py"
Task: "T003 - Update exports in inventory_monitor/core/__init__.py"
```

## Parallel Example: US2 + US3

```bash
# After Foundational phase, US2 and US3 can run in parallel:
# US2 modifies: face.py, app.py, gui.py
# US3 modifies: box.py, app.py, gui.py
# (app.py and gui.py overlap — run sequentially or coordinate carefully)
# Safest: run US2 first, then US3
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T007)
3. Complete Phase 3: User Story 1 (T008-T011)
4. **STOP and VALIDATE**: Walk a person through the door. Verify events.jsonl has entry with identity=null followed by identity_resolved event.
5. Deploy if event logging is the critical fix.

### Incremental Delivery

1. Setup + Foundational → Config and event infrastructure ready
2. US1 (T008-T011) → Zero missed events → Deploy (MVP!)
3. US4 (T012-T015) → Faster recognition → Deploy
4. US2 (T016-T022) → Self-learning faces → Deploy
5. US3 (T023-T029) → Self-learning boxes → Deploy
6. Polish (T030-T032) → Final validation

### Recommended Execution Order

For a single developer working sequentially:

1. T001 → T002 → T003 (Setup — 3 tasks)
2. T004 → T005 → T006 → T007 (Foundational — 4 tasks)
3. T008 → T009 → T010 → T011 (US1 — 4 tasks) **← MVP checkpoint**
4. T012 → T013 → T014 → T015 (US4 — 4 tasks)
5. T016 → T017 → T018 → T019 → T020 → T021 → T022 (US2 — 7 tasks)
6. T023 → T024 → T025 → T026 → T027 → T028 → T029 (US3 — 7 tasks)
7. T030 → T031 → T032 (Polish — 3 tasks)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- All tasks modify existing files — no new source files are created
- 6 files modified: config.py, event_manager.py, app.py, face.py, box.py, gui.py
- Commit after each phase or logical group of tasks
- Stop at any checkpoint to validate story independently
