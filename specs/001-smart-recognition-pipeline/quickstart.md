# Quickstart: Smart Recognition Pipeline

**Branch**: `001-smart-recognition-pipeline` | **Date**: 2026-02-05

## What Changed

This feature adds three capabilities to the CCTV Inventory Monitor:

1. **Guaranteed event logging** — Every door crossing is logged
   immediately, even when face recognition hasn't resolved yet. A
   follow-up event is sent when the identity is resolved.
2. **Self-learning faces** — Unknown faces are automatically collected
   and optionally auto-registered. Operators can correct labels via
   the existing GUI.
3. **Smarter recognition** — Face recognition prioritizes people
   approaching the door, so identities resolve faster.
4. **Self-learning boxes** — High-confidence box detections are
   auto-verified for training data collection.

## Configuration

Add these fields to your `config.json`:

```json
{
  "detection": {
    "auto_register": false,
    "auto_register_pattern": "Person_{}",
    "priority_recognition": true,
    "identity_resolve_timeout": 30.0
  },
  "training": {
    "auto_verify_threshold": 0.8,
    "retrain_min_samples": 500
  }
}
```

| Field                      | Default        | Description                              |
|----------------------------|----------------|------------------------------------------|
| `auto_register`            | `false`        | Auto-register unknown faces              |
| `auto_register_pattern`    | `"Person_{}"` | Label pattern for auto-registered faces  |
| `priority_recognition`     | `true`         | Prioritize face recognition near door    |
| `identity_resolve_timeout` | `30.0`         | Seconds before deferred event expires    |
| `auto_verify_threshold`    | `0.8`          | Min confidence for auto-verified samples |
| `retrain_min_samples`      | `500`          | Min verified samples for retraining      |

## Usage

### Automatic event logging (no action needed)

Once updated, every door crossing generates an API event immediately.
If the face is unrecognized, the event has `identity=null`. When the
face is later recognized, an `identity_resolved` event is sent
automatically.

### Enable auto-registration

Set `"auto_register": true` in `config.json`. New faces will be
registered automatically as "Person_001", "Person_002", etc.

To correct a label:
1. Press `C` in the GUI to enter correction mode
2. Use `Tab` to select the person
3. Type the correct name and press `Enter`

### Box model retraining

When training data collection is enabled (`"training.enabled": true`):
1. High-confidence detections are auto-verified
2. When enough samples accumulate, press `M` in the GUI to trigger
   retraining (runs in background)
3. The new model is loaded automatically when retraining completes

## New Keyboard Shortcuts

| Key | Action                                              |
|-----|-----------------------------------------------------|
| `M` | Trigger box model retraining (when samples ready)   |

All existing shortcuts remain unchanged.

## API Changes

A new event type `identity_resolved` is sent when a previously
unidentified person's face is recognized:

```json
{
  "event_type": "identity_resolved",
  "track_id": 42,
  "identity": "John Doe",
  "details": {
    "original_event_type": "entered",
    "original_timestamp": "2026-02-05T14:30:00",
    "resolution_delay_ms": 5665
  }
}
```

If auto-registration is enabled, a `face_auto_registered` event is
also sent for each new auto-registered face.

## Verification

To verify the feature is working:

1. **Event logging**: Check `logs/events.jsonl` — every door crossing
   should have an entry, even with `identity: null`. Look for
   `identity_resolved` events following them.
2. **Auto-registration**: Check `data/auto_registrations.jsonl` for
   auto-registered faces.
3. **Priority recognition**: Watch the GUI — persons approaching the
   door should be identified faster than stationary persons.
4. **Training data**: Check `training_queue/manifest.jsonl` for
   entries with `"verified": true`.
