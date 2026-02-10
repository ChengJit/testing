# Research: Body Re-Identification Pipeline

**Feature**: 002-body-reid-pipeline
**Date**: 2026-02-10

## Body Re-Identification Approach

### Decision: Hybrid Color Histogram + OSNet

**Rationale**:
- Color histogram provides instant, lightweight baseline (~40-50% accuracy, <1ms)
- OSNet adds robust deep features (~70-80% accuracy, 2-3ms with TensorRT)
- Combined approach achieves target 70%+ body-only accuracy
- Both methods fit within Jetson Orin Nano constraints

**Alternatives Considered**:

| Option | Accuracy | Latency | VRAM | Decision |
|--------|----------|---------|------|----------|
| Color histogram only | 40-50% | <1ms | 0 | Too low accuracy for production |
| OSNet only | 70-80% | 2-3ms | 80MB | Good, but lacks fallback |
| FastReID full | 80-85% | 5-8ms | 200MB+ | Too heavy for 15 FPS target |
| **Hybrid (chosen)** | 75-85% | 3-4ms | 80MB | Best balance |

### OSNet Model Selection

**Decision**: Use `osnet_x0_25` variant

**Rationale**:
- Smallest OSNet variant at 2.2MB
- 128D embedding dimension (same as face embeddings)
- TensorRT FP16 reduces to 0.6MB, 2-3ms inference
- Proven accuracy on person ReID benchmarks

**Source**: [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)

## Embedding Storage Format

### Decision: NPZ + JSON (matching face recognition pattern)

**Rationale**:
- Consistent with existing `known_embeddings.npz` for faces
- Proven pattern in codebase
- Efficient NumPy serialization for embeddings
- JSON for metadata (last_seen, sample_count)

**Structure**:
```
data/
├── face_embeddings.npz      # Existing face embeddings
├── body_embeddings.npz      # NEW: Body embeddings (128D)
├── body_histograms.npz      # NEW: Color histograms (1152D)
└── person_profiles.json     # NEW: Combined metadata
```

## Threading Architecture

### Decision: Dedicated Capture + Recognition threads

**Rationale**:
- Decouples capture from recognition to guarantee event logging
- Recognition can queue up without blocking main pipeline
- Matches spec requirement for <1 second event logging

**Architecture**:
```
Main Thread:        Video capture, display (if GUI)
                          ↓
Capture Thread:     Door crossing detection, body crop extraction
                          ↓ (queue)
Recognition Thread: Face recognition → Body ReID → API updates
```

## Score Fusion Strategy

### Decision: Face-priority with body validation

**Fusion Formula**:
```python
if face_score > 0.45:
    combined = face_score * 0.7 + body_score * 0.3
else:
    combined = body_score  # Body-only when face unavailable
```

**Threshold Matrix**:

| Face Score | Body Score | Action |
|------------|------------|--------|
| >0.75 | any | Use face identity (high confidence) |
| 0.45-0.75 | >0.6 | Boost confidence, use face identity |
| 0.45-0.75 | <0.6 | Use face with caution flag |
| <0.45 | >0.7 | Use body-only identity |
| <0.45 | 0.5-0.7 | Hold for more samples |
| <0.45 | <0.5 | Unknown - queue for auto-registration |

## Closest Match Algorithm

### Decision: Top-K with margin check

**Rationale**:
- Return best match if significantly better than second-best
- Flag for review if top-2 scores are within 10%
- Prevents false auto-assignment with ambiguous matches

**Implementation**:
```python
def find_closest_match(query_embedding, known_embeddings, threshold=0.5):
    scores = cosine_similarity(query_embedding, known_embeddings)
    sorted_idx = np.argsort(scores)[::-1]

    best_score = scores[sorted_idx[0]]
    if best_score < threshold:
        return None, 0, False  # No match

    second_score = scores[sorted_idx[1]] if len(sorted_idx) > 1 else 0
    margin = best_score - second_score

    needs_review = margin < 0.1  # Within 10%
    return names[sorted_idx[0]], best_score, needs_review
```

## Run Mode Unification

### Decision: Shared pipeline with display abstraction

**Rationale**:
- Same capture/recognition pipeline for --gui and --headless
- GUI mode adds display thread, headless skips it
- All logging and API calls identical in both modes

**Implementation**:
```python
# run_monitor.py structure
if args.headless:
    pipeline = MonitorPipeline(config)
    pipeline.run()  # Blocking, no display
else:
    pipeline = MonitorPipeline(config)
    gui = MonitorGUI(pipeline)  # Wraps pipeline with display
    gui.run()
```

## Resource Budget Analysis

### Current System:
- YOLO detection: ~300MB VRAM, 2-3ms
- InsightFace: ~400MB VRAM, 20-40ms
- ByteTracker: <50MB, <1ms
- **Total**: ~700MB VRAM

### With Body ReID:
- Add OSNet (TensorRT): +80MB VRAM, +2-3ms
- Add color histogram: +0MB, +<1ms
- **New Total**: ~780MB VRAM (well within 8GB)

### FPS Impact:
- Current: 15-20 FPS AI processing
- With body ReID: 14-18 FPS (acceptable, meets 15 FPS target)

## Dependencies

### No New Dependencies Required

Existing stack sufficient:
- OpenCV: Color histogram extraction
- NumPy: Embedding operations, storage
- PyTorch: OSNet model loading
- ONNX Runtime: Optional optimization

### OSNet Model File
- Download: `osnet_x0_25.pth` (2.2MB) from deep-person-reid
- Optional: Convert to TensorRT for best Jetson performance
