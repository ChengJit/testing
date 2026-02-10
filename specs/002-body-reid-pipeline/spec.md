# Feature Specification: Full-Body Person Re-Identification Pipeline

**Feature Branch**: `002-body-reid-pipeline`
**Created**: 2026-02-10
**Status**: Draft
**Input**: User description: "Full-body person re-identification with decoupled capture/recognition threading. Two-thread architecture: Thread 1 captures full body crops at door crossing moments and logs events immediately with track_id. Thread 2 processes recognition queue - tries face first, falls back to body/clothing matching when face not visible, cross-references front/back views, finds closest match for unknowns, auto-registers new persons, sends IDENTITY_RESOLVED to API. Rework run_monitor.py to support both --gui and --headless modes with same pipeline. Body embeddings stored alongside face embeddings for cross-referencing 'whose back is that'."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Identify People Facing Away from Camera (Priority: P1)

A warehouse operator observes that some people walk through the door with their back to the camera (e.g., pushing a cart, carrying boxes). The current face-only recognition fails to identify them. The system should recognize these people by their body appearance (clothing, body shape) by cross-referencing with previously captured front views of the same person.

**Why this priority**: This is the core value proposition. Without body-based recognition, a significant percentage of door crossings go unidentified, creating gaps in the audit trail.

**Independent Test**: Can be verified by having a known person walk through the door facing away. The system should match their body appearance to their previously stored front-view profile and identify them correctly.

**Acceptance Scenarios**:

1. **Given** a person has been previously identified (front view with face), **When** they cross the door with their back to the camera, **Then** the system MUST identify them by matching body appearance within 5 seconds.
2. **Given** a person is wearing the same clothing as their last visit, **When** they enter facing away, **Then** the system MUST achieve at least 70% match confidence based on body features.
3. **Given** a person's body appearance matches multiple known individuals, **When** confidence scores are close (within 10%), **Then** the system MUST flag for operator review rather than auto-assigning identity.

---

### User Story 2 - Immediate Event Logging with Deferred Identity (Priority: P1)

An operator needs every door crossing event to be logged immediately for audit purposes, even before identity is resolved. The system should capture the moment of crossing, log the event with a track ID, and asynchronously resolve the identity without blocking the capture pipeline.

**Why this priority**: Critical for audit completeness. Events must never be missed due to slow recognition processing.

**Independent Test**: Can be verified by having someone walk quickly through the door. The event should appear in logs within 1 second of crossing, with identity resolved and updated within 10 seconds.

**Acceptance Scenarios**:

1. **Given** a person crosses the door line, **When** recognition has not yet completed, **Then** the system MUST log the event within 1 second with track_id and identity=null.
2. **Given** an event was logged with unknown identity, **When** the recognition thread resolves identity (face or body), **Then** the system MUST send IDENTITY_RESOLVED to the API within 10 seconds.
3. **Given** the recognition queue has 10+ pending items, **When** new crossings occur, **Then** capture and logging MUST NOT be blocked or delayed.

---

### User Story 3 - Multi-Modal Recognition with Fallback (Priority: P2)

When a person crosses the door, the system should try multiple recognition methods in order of reliability: face recognition first (most accurate), then body/clothing matching (when face not visible), then closest match among known persons.

**Why this priority**: Maximizes identification rate by using all available visual information.

**Independent Test**: Can be verified by testing recognition with face visible, face partially occluded, and face completely hidden. All three scenarios should result in identification (with varying confidence levels).

**Acceptance Scenarios**:

1. **Given** a person's face is clearly visible, **When** crossing the door, **Then** the system MUST use face recognition as primary identification method.
2. **Given** a person's face is not visible (turned away, occluded), **When** crossing the door, **Then** the system MUST fall back to body appearance matching.
3. **Given** neither face nor body produces an exact match, **When** there is a similar person in the database (above minimum threshold), **Then** the system MUST suggest the closest match with confidence score.
4. **Given** no match is found above minimum threshold, **When** sufficient samples are collected, **Then** the system MUST auto-register as a new person.

---

### User Story 4 - Unified Pipeline for GUI and Headless Modes (Priority: P2)

The system should work identically whether running with a display (--gui) or without (--headless). Operators should be able to deploy the same recognition pipeline on display-equipped workstations and headless edge devices.

**Why this priority**: Enables flexible deployment across different hardware configurations without code changes.

**Independent Test**: Can be verified by running the same recognition scenario in both modes and comparing the logs/API calls - they should be identical.

**Acceptance Scenarios**:

1. **Given** the system is started with --gui flag, **When** a person crosses the door, **Then** the capture/recognition pipeline MUST function identically to headless mode.
2. **Given** the system is started with --headless flag, **When** a person crosses the door, **Then** all events MUST be logged and API calls MUST be sent without requiring a display.
3. **Given** either mode is running, **When** checking logs and API payloads, **Then** the event data MUST be identical for the same scenario.

---

### User Story 5 - Front/Back View Cross-Referencing (Priority: P3)

The system should build a complete visual profile of each person by linking their front view (face) with their back view (body from behind). This allows the system to recognize "whose back is that" by referencing previously captured front views.

**Why this priority**: Enhances recognition accuracy over time as more views are collected.

**Independent Test**: Can be verified by having a new person enter (face visible), then exit (back to camera). The system should link both views to the same identity.

**Acceptance Scenarios**:

1. **Given** a new person enters facing the camera, **When** they exit with back to camera within the same session, **Then** the system MUST link both views to the same track_id/identity.
2. **Given** a person has both front and back views stored, **When** they return on a different day, **Then** the system MUST recognize them from either view.
3. **Given** a person's front/back views are linked, **When** an operator views their profile, **Then** both views MUST be displayed together.

---

### Edge Cases

- What happens when two people wearing similar clothing cross at the same time? The system should use track continuity and any visible distinguishing features; if ambiguous, flag both for review.
- How does the system handle a person who changes clothing between visits? Face recognition takes priority; body recognition is per-session until face confirms identity.
- What happens when the camera view is partially blocked? The system should use whatever body portion is visible for matching; if insufficient, log as unidentified.
- How does the system handle identical twins? Rely on clothing differences within a session; flag low-margin matches for operator review.
- What happens when lighting conditions change dramatically? Body color matching should use relative color ratios rather than absolute values.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST capture full-body crops (not just face) at the moment of door crossing.
- **FR-002**: System MUST operate with two decoupled threads: one for capture/logging, one for recognition/training.
- **FR-003**: System MUST log door-crossing events within 1 second, regardless of recognition status.
- **FR-004**: System MUST attempt face recognition first, falling back to body recognition when face is not visible.
- **FR-005**: System MUST store body appearance embeddings alongside face embeddings for each known person.
- **FR-006**: System MUST cross-reference front and back views to link them to the same identity.
- **FR-007**: System MUST find and suggest the closest matching known person when exact match is not found.
- **FR-008**: System MUST auto-register truly new persons after collecting sufficient samples.
- **FR-009**: System MUST send IDENTITY_RESOLVED events to the API when identity is determined after initial logging.
- **FR-010**: System MUST support both --gui and --headless modes with identical recognition pipeline behavior.
- **FR-011**: System MUST maintain recognition processing even when display thread is blocked or unavailable.
- **FR-012**: System MUST queue recognition tasks and process them without blocking capture.

### Key Entities

- **PersonProfile**: A complete identity record containing face embeddings (if available), body embeddings (front/back views), display name, registration timestamp, and confidence history.
- **CaptureEvent**: A queued item containing track_id, timestamp, full_body_crop, face_crop (if detected), event_type (entry/exit), box_count, and processing_status.
- **BodyEmbedding**: A feature vector representing body appearance (clothing color histogram, body shape features, texture patterns) associated with a PersonProfile.
- **RecognitionResult**: The output of the recognition thread containing matched identity (or null), confidence score, match_method (face/body/closest), and any flags (low_margin, needs_review).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 95% of door-crossing events are logged within 1 second of occurrence, regardless of recognition status.
- **SC-002**: Identity is resolved for 85% of crossings where the person is already known (via face or body).
- **SC-003**: Body-only recognition achieves at least 70% accuracy for persons whose front view was previously captured.
- **SC-004**: Recognition processing does not cause any dropped frames or capture delays (capture thread maintains 15+ FPS).
- **SC-005**: IDENTITY_RESOLVED events are sent to the API within 10 seconds of initial event logging.
- **SC-006**: System performs identically in --gui and --headless modes (same events, same API calls, same log entries).
- **SC-007**: Front/back view cross-referencing correctly links 90% of same-person entries within a single session.
- **SC-008**: Auto-registration of new persons occurs within 3 visits without manual intervention.

### Assumptions

- The camera position provides a clear view of both approaching persons (front) and departing persons (back).
- Persons typically wear consistent clothing within a single day/session.
- The existing YOLO person detection provides reliable full-body bounding boxes.
- Body appearance features (clothing, shape) provide sufficient discriminative power for the typical number of tracked individuals (1-20 regular visitors).
- The Jetson Orin Nano has sufficient compute resources to run body embedding extraction alongside face recognition without dropping below 15 FPS.
