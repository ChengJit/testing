# Feature Specification: Smart Recognition Pipeline

**Feature Branch**: `001-smart-recognition-pipeline`
**Created**: 2026-02-05
**Status**: Draft
**Input**: User description: "Self-learning face and box recognition with minimal user correction, fix recognition delay causing missed API event logging for entries and exits"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Log All Entries and Exits Even When Face Is Unrecognized (Priority: P1)

An operator monitoring a warehouse door notices that people visibly walk in and out on the camera feed, but the system does not log some of these events to the API. The root cause is that the face recognition result arrives after the person has already crossed the door line, so the event fires with an unknown identity and is either not sent or lost. The system MUST log every door-crossing event regardless of whether a face has been recognized, and MUST retroactively update the event with the identity once recognition completes.

**Why this priority**: This is the most critical bug. If events are not logged, the entire monitoring system fails its core purpose. The API consumers (ops-portal) never receive the entry/exit record, making the inventory audit incomplete.

**Independent Test**: Can be verified by having a person walk through the door while face recognition is still processing. The system should immediately log an event (with identity "Unknown" or track ID) and then send a follow-up update with the resolved identity.

**Acceptance Scenarios**:

1. **Given** a tracked person crosses the door line, **When** face recognition has not yet resolved their identity, **Then** the system MUST immediately log and send an entry/exit event with `identity=null` and a valid `track_id`.
2. **Given** an event was logged with unknown identity, **When** face recognition later resolves the identity for that track, **Then** the system MUST send an identity-update event to the API linking the `track_id` to the resolved identity.
3. **Given** a person enters and exits quickly (under 5 seconds), **When** face recognition never resolves, **Then** both the entry and exit events MUST still be logged with the track ID, and a flag indicating the identity was unresolved.
4. **Given** the API is temporarily unreachable, **When** events are generated, **Then** events MUST be queued locally and retried when connectivity is restored.

---

### User Story 2 - Self-Learning Face Recognition (Priority: P2)

A new employee arrives at the warehouse. The system does not recognize them and labels them as "Unknown". Over the course of their first few visits, the system automatically collects face samples. After sufficient samples are gathered, the system prompts the operator (or auto-registers if configured) to assign a name. On subsequent visits the person is recognized without any manual intervention.

**Why this priority**: Reducing manual correction effort is essential for practical deployment. Without self-learning, every new person requires manual registration, which operators forget or skip.

**Independent Test**: Can be verified by introducing an unregistered person to the camera. After a configurable number of sightings, the system should either auto-register them or prompt the operator for a name.

**Acceptance Scenarios**:

1. **Given** an unrecognized person appears on camera multiple times, **When** the system has collected enough face samples (configurable threshold, default: 5 samples), **Then** the system MUST either auto-register with a generated label or prompt the operator to assign a name.
2. **Given** an operator assigns a name to an auto-collected face, **When** the person reappears, **Then** the system MUST recognize them within 3 seconds of face visibility with confidence above the recognition threshold.
3. **Given** auto-registration is enabled, **When** an unknown face accumulates sufficient samples without operator intervention, **Then** the system MUST automatically register the face with a generated label (e.g., "Person_001") and begin recognizing them immediately.
4. **Given** a face was auto-registered with a generated label, **When** an operator later corrects the name via the GUI, **Then** the system MUST update the label, retrain embeddings, and use the corrected name going forward for both new events and display.

---

### User Story 3 - Self-Learning Box Detection (Priority: P3)

The box detector occasionally misses boxes or produces false positives. The system collects training data in the background (already partially implemented) but requires manual triggering and verification. The system should continuously collect high-confidence positive and negative samples, and when enough verified data accumulates, support triggering a retraining cycle for the custom YOLO box model.

**Why this priority**: Improving box detection accuracy over time reduces false inventory audit entries. This builds on existing training data collection but makes it more autonomous and useful.

**Independent Test**: Can be verified by enabling training data collection and observing that the system automatically curates positive and negative samples. A retraining trigger should be invocable when sufficient samples accumulate.

**Acceptance Scenarios**:

1. **Given** training data collection is enabled, **When** the system detects boxes with high confidence (above a configurable threshold), **Then** it MUST automatically save the sample as a verified positive training example.
2. **Given** the system has accumulated a configurable number of verified samples (default: 500), **When** an operator triggers retraining, **Then** the system MUST retrain the custom box detection model using the collected data.
3. **Given** training data collection is active, **When** a tracked person without detected boxes passes through, **Then** the system MUST periodically save negative samples at the configured ratio.

---

### User Story 4 - Reduce Face Recognition Latency (Priority: P2)

Operators observe that face recognition takes several seconds after a person appears before a name is displayed. This delay causes the identity to be unavailable when the person crosses the door line. The system should prioritize face recognition for persons approaching the door zone so identities are resolved before the door crossing event fires.

**Why this priority**: Same priority as self-learning because it directly addresses the root cause of missed identities on events. Even with deferred logging (US1), faster recognition means fewer deferred updates and cleaner data.

**Independent Test**: Can be verified by measuring the time from first person detection to identity resolution. The target is that identity is resolved before the person crosses the door line in 80% of normal-speed entries.

**Acceptance Scenarios**:

1. **Given** a person is detected approaching the door zone, **When** their face is visible, **Then** the system MUST prioritize face recognition for that track over tracks that are stationary or moving away from the door.
2. **Given** a person walks at normal speed (2-4 seconds from detection to door crossing), **When** their face is visible for at least 1 second, **Then** the system MUST resolve their identity before they cross the door line at least 80% of the time.
3. **Given** multiple people are in frame simultaneously, **When** one is approaching the door and another is stationary inside, **Then** the approaching person's face recognition MUST be processed first.

---

### Edge Cases

- What happens when two people cross the door at the same time with unresolved identities? Both events MUST be logged with their respective track IDs, and identity updates MUST be sent independently for each.
- How does the system handle a person whose face is never visible (e.g., wearing a mask or turned away)? The event MUST still be logged with `identity=null` and the track ID. No identity-update event is sent.
- What happens if auto-registered faces have near-identical embeddings (e.g., twins)? The system MUST flag low-margin matches for operator review rather than silently misidentifying.
- What happens when the face recognizer retrains while the system is running? Retraining MUST NOT block the main processing pipeline; new embeddings MUST be swapped in atomically.
- What happens if the API receives a deferred identity-update for a track_id it has never seen? The API MUST accept the update and store it; reconciliation is the API consumer's responsibility.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST log every door-crossing event immediately upon state machine transition, regardless of identity resolution status.
- **FR-002**: System MUST send a follow-up identity-update event to the API when a previously unknown track_id receives a face recognition match.
- **FR-003**: System MUST support deferred event enrichment: events logged with unknown identity MUST be retroactively updated in local logs (CSV, JSON) when identity is resolved.
- **FR-004**: System MUST auto-collect face samples for unrecognized persons and support both auto-registration (with generated labels) and operator-prompted registration.
- **FR-005**: System MUST allow operators to correct auto-generated labels via the existing GUI correction flow, triggering automatic embedding retraining.
- **FR-006**: System MUST prioritize face recognition for tracks approaching the door zone over tracks in other areas.
- **FR-007**: System MUST support configurable auto-registration behavior: off (operator prompt only), on with generated labels, or on with configurable naming patterns.
- **FR-008**: System MUST continuously collect high-confidence box detection samples as verified training data when training collection is enabled.
- **FR-009**: System MUST support triggering box model retraining from collected verified samples (manual trigger acceptable).
- **FR-010**: System MUST NOT block the main processing loop during face embedding retraining or box model retraining.
- **FR-011**: System MUST queue API events locally when the API is unreachable and retry when connectivity returns.

### Key Entities

- **DeferredEvent**: An entry/exit event logged before identity resolution, containing track_id, timestamp, event_type, box_count, and a pending identity field. Becomes enriched when identity resolves.
- **AutoRegistration**: A record of an automatically registered face, containing generated label, track_id, sample count, registration timestamp, and a flag indicating whether the operator has reviewed/corrected it.
- **RecognitionPriority**: A weighting assigned to each active track indicating how urgently face recognition should process it, based on proximity to door zone and movement direction.
- **VerifiedTrainingSample**: A training data entry that has been automatically verified by high-confidence detection thresholds, ready for model retraining without manual review.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of door-crossing events are logged and sent to the API, regardless of identity resolution status (zero missed events).
- **SC-002**: 80% of identities are resolved before door crossing for persons whose face is visible for at least 1 second while approaching.
- **SC-003**: Operators need to manually register or correct fewer than 20% of recurring visitors after the system has been running for one week.
- **SC-004**: Face recognition for new (previously unseen) persons automatically collects sufficient samples within 3 visits without operator action.
- **SC-005**: Deferred identity updates are sent to the API within 10 seconds of identity resolution.
- **SC-006**: Retraining face embeddings or box detection models does not cause any dropped frames or processing pipeline stalls.
- **SC-007**: The system produces at least 50 verified training samples per day of operation when training collection is enabled.

### Assumptions

- The existing API endpoint supports receiving identity-update events (a new event type or update mechanism). If not, the API will need a minor extension.
- Normal walking speed through the monitored door is 2-4 seconds from first detection to door crossing.
- The Jetson Orin Nano has sufficient GPU headroom to prioritize face recognition without reducing overall AI FPS below 15.
- Auto-registration with generated labels is acceptable as a default behavior; operators can be trained to correct labels periodically.
