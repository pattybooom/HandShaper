# HandShaperr - Copilot Instructions

## Project Overview

HandShaperr classifies geometric shapes formed by hands/fingers (square, triangle, circle, star, etc.) using MediaPipe hand landmarks in real-time. This is NOT sign language or gesture recognition—it's **hand-formed shape classification**.

## Architecture

```
data_collection.py   # Webcam → MediaPipe → normalized landmarks → CSV/JSONL/NPZ
train.py             # Load dataset → train PyTorch MLP → save best.pt + metrics
infer_realtime.py    # Webcam → MediaPipe → model → predicted shape + confidence
ws_server.py         # WebSocket server for browser-based inference
model.py             # HandShapeClassifier: MLP with LayerNorm, GELU, dropout
dataset.py           # Multi-format loader (JSONL/CSV/NPZ), feature extraction, splits
utils.py             # Normalization, standardization, prediction smoothing
dataset/             # Session folders with samples.csv, samples.jsonl, samples.npz
runs/                # Training runs with best.pt, label_map.json, confusion_matrix.png
web/                 # Browser frontend: index.html, app.js, styles.css
```

**Data Flow:**
1. **Collection:** Camera → MediaPipe (21×3×2 landmarks) → normalize (wrist-origin, scale) → save
2. **Training:** Load dataset → 128-dim features (landmarks + hand-present flags) → MLP → cross-entropy
3. **Inference (Python):** Camera → MediaPipe → same normalization → standardize → model → smoothed prediction
4. **Inference (Web):** Browser MediaPipe JS → WebSocket → ws_server.py → model → prediction → Three.js 3D shape

## Dependencies

```bash
pip install opencv-python mediapipe numpy torch scikit-learn matplotlib tqdm websockets
```

## Commands

```bash
# Data collection
python data_collection.py --camera 0 --min_confidence 0.7 --two_hands True

# Training
python train.py --data_dir ./dataset --out_dir ./runs --epochs 50 --hidden 256

# Real-time inference (Python)
python infer_realtime.py --checkpoint ./runs/run1/best.pt --smoothing_window 5

# Web interface (run both commands)
python ws_server.py --checkpoint ./runs/run1/best.pt --port 8000
# Then open web/index.html via local server: python -m http.server 8080
```

## Data Format

- **Features:** 128 values = 21×3×2 landmarks + 2 hand-present flags
- **Normalization:** Translate to wrist origin, scale by wrist→middle_mcp distance
- **Standardization:** Z-score using training set mean/std (saved to feature_stats.json)

## Key Patterns

- `normalize_landmarks()`: Position/scale invariance (in utils.py and data_collection.py)
- `extract_features()`: Consistent 128-dim vector for train and inference
- `PredictionSmoother`: Moving average over N frames for stable predictions
- Multi-format dataset loading: Auto-discovers JSONL, CSV, NPZ files recursively
- `--split_by_session`: Prevents data leakage by splitting on session_id

## Web Frontend

- **MediaPipe Hands JS** runs in browser, sends landmarks via WebSocket
- **ws_server.py** receives landmarks, runs PyTorch model, returns predictions
- **Three.js** renders 3D shapes (square=wireframe box, triangle=cone, circle=torus, heart=extruded heart)
- Client-side smoothing via majority vote over configurable window

## Model Architecture

```python
HandShapeClassifier(
    input_dim=128,      # 21*3*2 landmarks + 2 mask flags
    hidden_dim=256,     # Configurable via --hidden
    num_classes=N,      # Auto-detected from labels
    dropout=0.2,        # LayerNorm + GELU + Dropout
)
```

## Training Outputs

- `best.pt`: Model weights, optimizer state, label_map, feature_stats, config
- `confusion_matrix.png`: Visual evaluation on test set
- `history.json`: Per-epoch train/val loss and accuracy
- `test_results.json`: Final metrics and classification report
