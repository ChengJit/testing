# Quickstart: Body Re-Identification Pipeline

**Feature**: 002-body-reid-pipeline
**Date**: 2026-02-10

## Overview

This feature adds full-body person re-identification to the CCTV Inventory Monitor, enabling identification of people even when facing away from the camera.

## Key Changes

### New Files
- `inventory_monitor/detectors/body.py` - Body ReID (OSNet + color histogram)
- `inventory_monitor/core/recognition_worker.py` - Background recognition thread
- `inventory_monitor/core/capture_worker.py` - Background capture thread

### Modified Files
- `inventory_monitor/app.py` - Integrate dual-thread pipeline
- `inventory_monitor/config.py` - Add body ReID config fields
- `inventory_monitor/core/event_manager.py` - Add match_method field
- `run_monitor.py` - Unify --gui and --headless modes

## Configuration

New config fields in `config.json`:

```json
{
  "detection": {
    "body_reid_enabled": true,
    "body_model": "osnet_x0_25",
    "body_threshold": 0.6,
    "use_color_histogram": true,
    "face_body_weight": [0.7, 0.3]
  }
}
```

## Running

Both modes now use the same pipeline:

```bash
# GUI mode (with display)
python run_monitor.py --gui

# Headless mode (no display)
python run_monitor.py --headless
```

## Threading Model

```
┌─────────────────┐
│   Main Thread   │
│  (Video + GUI)  │
└────────┬────────┘
         │ frames
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Capture Worker  │────▶│ Recognition     │
│ - Door crossing │queue│   Worker        │
│ - Body crops    │     │ - Face ReID     │
│ - Immediate log │     │ - Body ReID     │
└─────────────────┘     │ - API updates   │
                        └─────────────────┘
```

## Testing

1. **Body-only recognition**: Have person walk through with back to camera
2. **Deferred identity**: Check logs for immediate event + later resolution
3. **Mode parity**: Run same scenario in --gui and --headless, compare logs

## Model Files

Download OSNet model (optional, for deep body ReID):
```bash
# From deep-person-reid repository
wget https://github.com/KaiyangZhou/deep-person-reid/releases/download/v0.2.0/osnet_x0_25_msmt17.pth
mv osnet_x0_25_msmt17.pth models/
```

Without OSNet, the system falls back to color histogram only (lower accuracy but zero additional dependencies).
