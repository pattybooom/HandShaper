# HandShaperr - Copilot Instructions

## Project Overview

HandShaperr classifies geometric shapes formed by hands/fingers (square, triangle, circle, star, etc.) using MediaPipe hand landmarks in real-time. This is NOT sign language or gesture recognition—it's **hand-formed shape classification**.

## Architecture

```
data_collection.py   # Webcam → MediaPipe → normalized landmarks → CSV/JSONL/NPZ
train.py             # Load dataset → train PyTorch MLP → save best.pt + metrics
infer_realtime.py    # Webcam → MediaPipe → model → predicted shape + confidence
model.py             # HandShapeClassifier: MLP with LayerNorm, GELU, dropout
dataset.py           # Multi-format loader (JSONL/CSV/NPZ), feature extraction, splits
utils.py             # Normalization, standardization, prediction smoothing
dataset/             # Session folders with samples.csv, samples.jsonl, samples.npz
runs/                # Training runs with best.pt, label_map.json, confusion_matrix.png
```

**Data Flow:**
1. **Collection:** Camera → MediaPipe (21×3×2 landmarks) → normalize (wrist-origin, scale) → save
2. **Training:** Load dataset → 128-dim features (landmarks + hand-present flags) → MLP → cross-entropy
3. **Inference:** Camera → MediaPipe → same normalization → standardize (saved stats) → model → smoothed prediction

## Dependencies

```bash
pip install opencv-python mediapipe numpy torch scikit-learn matplotlib tqdm
```

## Commands

```bash
# Data collection
python data_collection.py --camera 0 --min_confidence 0.7 --two_hands True

# Training
python train.py --data_dir ./dataset --out_dir ./runs --epochs 50 --hidden 256

# Real-time inference
python infer_realtime.py --checkpoint ./runs/run_XXXXXX/best.pt --smoothing_window 5
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
