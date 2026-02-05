<!--
  === Sync Impact Report ===
  Version change: 0.0.0 (new) -> 1.0.0
  Modified principles: N/A (initial adoption)
  Added sections:
    - Core Principles (6 principles)
    - Technical Constraints
    - Development Workflow
    - Governance
  Removed sections: None
  Templates requiring updates:
    - .specify/templates/plan-template.md ✅ compatible (Constitution Check section generic)
    - .specify/templates/spec-template.md ✅ compatible (no principle-specific references)
    - .specify/templates/tasks-template.md ✅ compatible (no principle-specific task types)
    - .specify/templates/checklist-template.md ✅ compatible (generic template)
    - .specify/templates/agent-file-template.md ✅ compatible (generic template)
  Follow-up TODOs: None
-->

# CCTV Inventory Monitor Constitution

## Core Principles

### I. Self-Learning & Recognition

All person and object recognition capabilities MUST support
incremental learning. The system MUST:

- Allow on-the-fly face registration and identity correction
  without restarting the application.
- Persist learned embeddings and training data so that
  recognition accuracy improves over time.
- Support automatic retraining of face embeddings when new
  samples are added (via GUI correction or registration).
- Collect training data for custom object detectors (e.g., box
  detector) when enabled, enabling supervised model improvement.

**Rationale**: A monitoring system that cannot adapt to new
personnel or objects loses value quickly. Continuous learning
reduces manual configuration and scales with the deployment.

### II. Real-Time Performance

Every processing pipeline MUST maintain frame-rate targets
suitable for live CCTV monitoring. Specifically:

- Display loop MUST sustain >= 30 FPS on the target platform.
- AI processing pipeline MUST sustain >= 15 FPS inference rate.
- Frame queues MUST drop stale frames rather than accumulate
  latency; queue depths MUST NOT exceed 2 frames.
- All heavy computation (detection, recognition) MUST run in
  background threads; the main loop MUST NOT block on inference.

**Rationale**: A monitoring system that lags behind real-time
cannot reliably track entries, exits, or carried objects. Latency
directly translates to missed or incorrect events.

### III. Edge-First Design

The system MUST be designed to run on resource-constrained edge
devices (primary target: NVIDIA Jetson Orin Nano, 8 GB VRAM).
This means:

- All default configurations MUST be tuned for the Jetson
  platform (TensorRT, FP16, nano-sized models).
- Memory management MUST include periodic GPU cache clearing
  and bounded queue sizes.
- Frame downscaling for AI processing MUST be configurable
  (ai_scale parameter) to trade accuracy for throughput.
- The system MUST function without cloud connectivity; API
  reporting is optional and MUST NOT block local operation.

**Rationale**: The deployment environment is a warehouse or
retail site with limited compute. Cloud-dependent designs
introduce latency, cost, and single points of failure.

### IV. Detection Reliability

Detection and tracking pipelines MUST produce consistent,
trustworthy results. This requires:

- Person tracking MUST use ByteTrack (or equivalent multi-object
  tracker) with configurable thresholds for confirmation.
- Box detection MUST support multiple detection strategies
  (custom YOLO model, GroundingDINO fallback) and MUST assign
  boxes to the nearest tracked person.
- Face recognition MUST implement a confidence-lock mechanism:
  identities below the lock threshold remain tentative; only
  high-confidence matches lock the identity for the track
  lifetime.
- State machine transitions (entered/exited) MUST require
  crossing a configurable door-line threshold; spurious
  oscillations MUST NOT generate duplicate events.

**Rationale**: False positives and missed events erode trust
in the inventory audit trail. Conservative thresholds with
human-correctable overrides balance automation and accuracy.

### V. Privacy & Security

All processing of video and biometric data MUST respect
privacy boundaries:

- Face images and embeddings MUST be stored locally on the
  edge device; they MUST NOT be transmitted to external
  services without explicit configuration (api.send_images).
- API credentials, RTSP URLs, and other secrets MUST NOT be
  committed to version control.
- Logging MUST NOT include raw face embeddings or full-frame
  images; structured event logs (entry/exit, box counts,
  identity labels) are permitted.
- The system MUST support headless operation (no GUI) to
  avoid exposing video feeds on shared displays.

**Rationale**: CCTV and face recognition systems handle
sensitive personal data. Local-first processing minimises
exposure; explicit opt-in for remote reporting ensures
informed consent.

### VI. Simplicity

Prefer the simplest solution that satisfies the requirement:

- Configuration MUST use a single flat JSON file loadable
  with dataclass defaults; avoid nested config frameworks.
- New features MUST NOT introduce additional runtime
  dependencies unless no existing dependency can fulfil the
  need.
- Abstractions MUST be justified by at least two concrete use
  cases in the codebase; single-use wrappers MUST be avoided.
- YAGNI applies: do not build for hypothetical future
  requirements. Extend when the need is demonstrated.

**Rationale**: The codebase runs on embedded hardware
maintained by a small team. Every unnecessary abstraction
increases cognitive load and deployment risk.

## Technical Constraints

The following constraints apply to all feature work:

- **Language**: Python 3.10+ (matching Jetson JetPack SDK).
- **Primary dependencies**: OpenCV, Ultralytics YOLO, InsightFace,
  NumPy. New dependencies require justification against this set.
- **Model format**: ONNX or TensorRT for inference; PyTorch
  weights accepted only for training and export.
- **Configuration**: All tuneable parameters MUST be exposed in
  `config.json` via the `Config` dataclass hierarchy. Hard-coded
  magic numbers MUST be extracted to config.
- **Logging**: Use Python `logging` module; MUST NOT use print()
  for operational output.
- **Platform**: Primary target is NVIDIA Jetson (aarch64 + CUDA).
  Desktop (x86_64 + CUDA or CPU) MUST remain functional for
  development and testing.

## Development Workflow

All contributors MUST follow this workflow:

1. **Branch per feature/fix**: Work on a dedicated branch; do not
   commit directly to `master`.
2. **Config before code**: Any new tuneable value MUST first be
   added to the relevant `*Config` dataclass with a sensible
   default before being used in application code.
3. **Test on target hardware**: Features involving detection
   thresholds, FPS targets, or memory usage MUST be validated on
   Jetson hardware (or documented as untested if hardware is
   unavailable).
4. **No secrets in commits**: RTSP URLs, API keys, and face
   images MUST be excluded via `.gitignore`. Review staged files
   before every commit.
5. **Incremental delivery**: Prefer small, self-contained changes
   that can be tested independently over large multi-file PRs.

## Governance

This constitution is the authoritative source of project
principles. In case of conflict between this document and any
other project documentation, this constitution prevails.

- **Amendments**: Any change to a Core Principle requires
  documentation of the rationale, an updated version number,
  and review by the project maintainer before merge.
- **Versioning**: The constitution follows semantic versioning:
  - MAJOR: Principle removed or fundamentally redefined.
  - MINOR: New principle or section added or materially expanded.
  - PATCH: Clarifications, wording, or non-semantic refinements.
- **Compliance review**: Feature specs and implementation plans
  MUST reference the Constitution Check section in plan-template
  and verify alignment with all six principles before work begins.

**Version**: 1.0.0 | **Ratified**: 2026-02-05 | **Last Amended**: 2026-02-05
