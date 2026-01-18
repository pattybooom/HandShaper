# HandShaperr - Copilot Instructions

## Project Overview

HandShaperr classifies geometric shapes formed by hands/fingers (square, triangle, circle, star, etc.) using MediaPipe hand landmarks in real-time. This is NOT sign language or gesture recognition—it's **hand-formed shape classification**.

## Architecture

```
data_collection.py   # Data collection: webcam → MediaPipe → normalized landmarks → CSV/JSONL/NPZ
dataset/             # Output: session folders with samples.csv, samples.jsonl, samples.npz, metadata.json
```

**Data Flow:** Camera → MediaPipe Hands (21 landmarks × 2 hands) → Normalize (wrist-origin, scale by hand size) → Save with label

## Dependencies

```bash
pip install opencv-python mediapipe numpy
```

## Running Data Collection

```bash
python data_collection.py
python data_collection.py --out_dir ./dataset --min_confidence 0.7 --two_hands False
python data_collection.py --record_every_n 3 --max_per_label 1000
```

**Controls:** `[L]` set label, `[R]` toggle recording, `[C]` capture one, `[N]` new session, `[Q]` quit

## Data Format

- **Landmarks:** 21 per hand, each (x, y, z), normalized to wrist origin and scaled by wrist→middle_mcp distance
- **Outputs:** CSV (flat), JSONL (structured), NPZ (numpy arrays for ML)
- **Per sample:** `sample_id`, `session_id`, `timestamp`, `label`, `hand1/hand2` data with `detected` mask

## Key Patterns

- `normalize_landmarks()`: Translate to wrist origin, scale by hand size—ensures position/size invariance
- `DatasetManager`: Incremental saves (crash-safe), session-based organization, metadata tracking
- `HandShapeCollector`: Main loop with `process_frame()` → `capture_sample()` → UI overlay

## Future Components

1. **Training script:** Load NPZ, train PyTorch/TensorFlow classifier on normalized landmarks
2. **Inference script:** Real-time prediction from webcam
3. **Visualization:** HTML page with webcam + predicted 3D shape rendering
