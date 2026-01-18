#!/usr/bin/env python3
"""
dataset.py - Data loading and preprocessing for HandShaperr
"""

import csv
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import normalize_landmarks, compute_feature_stats, standardize_features, save_json


# =============================================================================
# DATA LOADING FROM MULTIPLE FORMATS
# =============================================================================


def load_jsonl_file(path: Path) -> list[dict]:
    """Load samples from a JSONL file."""
    samples = []
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                warnings.warn(f"Skipping malformed JSON at {path}:{line_num}: {e}")
    return samples


def load_csv_file(path: Path) -> list[dict]:
    """Load samples from a CSV file."""
    samples = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, 1):
            try:
                sample = parse_csv_row(row)
                if sample is not None:
                    samples.append(sample)
            except Exception as e:
                warnings.warn(f"Skipping malformed CSV row at {path}:{row_num}: {e}")
    return samples


def parse_csv_row(row: dict) -> Optional[dict]:
    """Parse a CSV row into a sample dict matching JSONL format."""
    if 'label' not in row:
        return None

    # Extract landmarks for hand1
    hand1_landmarks = []
    for i in range(21):
        try:
            x = float(row.get(f'h1_lm{i}_x', 0))
            y = float(row.get(f'h1_lm{i}_y', 0))
            z = float(row.get(f'h1_lm{i}_z', 0))
            hand1_landmarks.append([x, y, z])
        except (ValueError, TypeError):
            hand1_landmarks.append([0.0, 0.0, 0.0])

    # Extract landmarks for hand2
    hand2_landmarks = []
    for i in range(21):
        try:
            x = float(row.get(f'h2_lm{i}_x', 0))
            y = float(row.get(f'h2_lm{i}_y', 0))
            z = float(row.get(f'h2_lm{i}_z', 0))
            hand2_landmarks.append([x, y, z])
        except (ValueError, TypeError):
            hand2_landmarks.append([0.0, 0.0, 0.0])

    # Get detection flags
    h1_detected = row.get('h1_detected', '1')
    h2_detected = row.get('h2_detected', '0')

    # Handle both string and numeric values
    if isinstance(h1_detected, str):
        h1_detected = h1_detected.lower() in ('1', 'true', 'yes')
    else:
        h1_detected = bool(h1_detected)

    if isinstance(h2_detected, str):
        h2_detected = h2_detected.lower() in ('1', 'true', 'yes')
    else:
        h2_detected = bool(h2_detected)

    return {
        'label': row['label'],
        'session_id': row.get('session_id', 'unknown'),
        'hand1': {
            'landmarks_normalized': hand1_landmarks,
            'detected': h1_detected,
        },
        'hand2': {
            'landmarks_normalized': hand2_landmarks,
            'detected': h2_detected,
        },
    }


def load_npz_file(path: Path) -> list[dict]:
    """Load samples from an NPZ file."""
    samples = []
    try:
        data = np.load(path, allow_pickle=True)

        # Get arrays
        hand1_landmarks = data.get('hand1_landmarks')
        hand2_landmarks = data.get('hand2_landmarks')
        hand1_detected = data.get('hand1_detected')
        hand2_detected = data.get('hand2_detected')
        labels = data.get('labels')
        sample_ids = data.get('sample_ids', None)

        if hand1_landmarks is None or labels is None:
            warnings.warn(f"NPZ file {path} missing required arrays")
            return []

        n_samples = len(labels)

        for i in range(n_samples):
            # Extract session_id from sample_id if available
            session_id = 'unknown'
            if sample_ids is not None and i < len(sample_ids):
                sid = str(sample_ids[i])
                if '_' in sid:
                    session_id = '_'.join(sid.split('_')[:-1])

            sample = {
                'label': str(labels[i]),
                'session_id': session_id,
                'hand1': {
                    'landmarks_normalized': hand1_landmarks[i].tolist(),
                    'detected': bool(hand1_detected[i]) if hand1_detected is not None else True,
                },
                'hand2': {
                    'landmarks_normalized': hand2_landmarks[i].tolist() if hand2_landmarks is not None else [[0.0]*3]*21,
                    'detected': bool(hand2_detected[i]) if hand2_detected is not None else False,
                },
            }
            samples.append(sample)

    except Exception as e:
        warnings.warn(f"Error loading NPZ file {path}: {e}")

    return samples


def discover_and_load_dataset(data_dir: Path) -> list[dict]:
    """
    Recursively discover and load all supported dataset files.

    Args:
        data_dir: Root directory to search.

    Returns:
        List of all samples from all files.
    """
    all_samples = []

    # Find all supported files
    jsonl_files = list(data_dir.rglob("*.jsonl"))
    csv_files = list(data_dir.rglob("*.csv"))
    npz_files = list(data_dir.rglob("*.npz"))

    print(f"[INFO] Found {len(jsonl_files)} JSONL, {len(csv_files)} CSV, {len(npz_files)} NPZ files")

    # Load JSONL files
    for path in jsonl_files:
        samples = load_jsonl_file(path)
        print(f"  Loaded {len(samples)} samples from {path.name}")
        all_samples.extend(samples)

    # Load CSV files
    for path in csv_files:
        samples = load_csv_file(path)
        print(f"  Loaded {len(samples)} samples from {path.name}")
        all_samples.extend(samples)

    # Load NPZ files
    for path in npz_files:
        samples = load_npz_file(path)
        print(f"  Loaded {len(samples)} samples from {path.name}")
        all_samples.extend(samples)

    return all_samples


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================


def extract_features(sample: dict, apply_normalization: bool = True) -> Optional[np.ndarray]:
    """
    Extract fixed-size feature vector from a sample.

    Features: [hand1_landmarks (63), hand2_landmarks (63), hand1_present (1), hand2_present (1)]
    Total: 128 features

    Args:
        sample: Sample dictionary with hand1/hand2 data.
        apply_normalization: Whether to normalize landmarks (should be False if already normalized).

    Returns:
        Feature vector of shape (128,) or None if extraction fails.
    """
    try:
        # Extract hand1
        hand1 = sample.get('hand1', {})
        hand1_lm = np.array(hand1.get('landmarks_normalized', [[0.0]*3]*21), dtype=np.float32)
        hand1_detected = float(hand1.get('detected', False))

        # Extract hand2
        hand2 = sample.get('hand2', {})
        hand2_lm = np.array(hand2.get('landmarks_normalized', [[0.0]*3]*21), dtype=np.float32)
        hand2_detected = float(hand2.get('detected', False))

        # Ensure correct shape
        if hand1_lm.shape != (21, 3):
            hand1_lm = np.zeros((21, 3), dtype=np.float32)
            hand1_detected = 0.0
        if hand2_lm.shape != (21, 3):
            hand2_lm = np.zeros((21, 3), dtype=np.float32)
            hand2_detected = 0.0

        # Apply normalization if needed (usually already normalized from collection)
        if apply_normalization and hand1_detected > 0.5:
            hand1_lm = normalize_landmarks(hand1_lm)
        if apply_normalization and hand2_detected > 0.5:
            hand2_lm = normalize_landmarks(hand2_lm)

        # Flatten and concatenate
        features = np.concatenate([
            hand1_lm.flatten(),  # 63 values
            hand2_lm.flatten(),  # 63 values
            [hand1_detected],    # 1 value
            [hand2_detected],    # 1 value
        ])

        return features.astype(np.float32)

    except Exception as e:
        warnings.warn(f"Feature extraction failed: {e}")
        return None


def build_label_mapping(samples: list[dict]) -> dict[str, int]:
    """
    Build deterministic label to index mapping.

    Args:
        samples: List of sample dictionaries.

    Returns:
        Dict mapping label strings to integer indices.
    """
    labels = sorted(set(s['label'] for s in samples))
    return {label: idx for idx, label in enumerate(labels)}


# =============================================================================
# PYTORCH DATASET
# =============================================================================


class HandShapeDataset(Dataset):
    """PyTorch Dataset for hand shape classification."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        stats: Optional[dict] = None,
    ):
        """
        Args:
            features: Array of shape (N, 128) with feature vectors.
            labels: Array of shape (N,) with integer class labels.
            stats: Optional dict with 'mean' and 'std' for standardization.
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.stats = stats

        # Apply standardization if stats provided
        if stats is not None:
            self.features = standardize_features(self.features, stats)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x, y


# =============================================================================
# DATA SPLITTING
# =============================================================================


def train_val_test_split(
    samples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_by_session: bool = False,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split samples into train/val/test sets.

    Args:
        samples: List of sample dictionaries.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        split_by_session: If True, split by session_id to avoid leakage.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_samples, val_samples, test_samples).
    """
    np.random.seed(seed)

    if split_by_session:
        return _split_by_session(samples, train_ratio, val_ratio, seed)
    else:
        return _stratified_split(samples, train_ratio, val_ratio, seed)


def _stratified_split(
    samples: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by label."""
    from sklearn.model_selection import train_test_split

    labels = [s['label'] for s in samples]

    # First split: train vs (val+test)
    train_samples, temp_samples = train_test_split(
        samples,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test (50/50 of remaining)
    temp_labels = [s['label'] for s in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples,
        train_size=val_ratio / (1 - train_ratio),
        stratify=temp_labels,
        random_state=seed,
    )

    return train_samples, val_samples, test_samples


def _split_by_session(
    samples: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split by session_id to avoid data leakage."""
    # Group by session
    sessions: dict[str, list[dict]] = {}
    for s in samples:
        sid = s.get('session_id', 'unknown')
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(s)

    # Shuffle session IDs
    session_ids = list(sessions.keys())
    np.random.shuffle(session_ids)

    # Split sessions
    n_sessions = len(session_ids)
    n_train = int(n_sessions * train_ratio)
    n_val = int(n_sessions * val_ratio)

    train_sessions = session_ids[:n_train]
    val_sessions = session_ids[n_train:n_train + n_val]
    test_sessions = session_ids[n_train + n_val:]

    # Collect samples
    train_samples = [s for sid in train_sessions for s in sessions[sid]]
    val_samples = [s for sid in val_sessions for s in sessions[sid]]
    test_samples = [s for sid in test_sessions for s in sessions[sid]]

    return train_samples, val_samples, test_samples


# =============================================================================
# MAIN DATA PIPELINE
# =============================================================================


def prepare_data(
    data_dir: Path,
    out_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_by_session: bool = False,
    seed: int = 42,
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader, dict, dict, int]:
    """
    Full data preparation pipeline.

    Args:
        data_dir: Directory containing dataset files.
        out_dir: Directory to save label_map.json and feature_stats.json.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        split_by_session: Whether to split by session.
        seed: Random seed.
        batch_size: Batch size for DataLoaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_map, feature_stats, num_classes).
    """
    # Load all data
    print("\n[INFO] Loading dataset...")
    samples = discover_and_load_dataset(data_dir)

    if len(samples) == 0:
        raise ValueError(f"No valid samples found in {data_dir}")

    print(f"[INFO] Total samples loaded: {len(samples)}")

    # Build label mapping
    label_map = build_label_mapping(samples)
    num_classes = len(label_map)
    print(f"[INFO] Found {num_classes} classes: {list(label_map.keys())}")

    # Split data
    print(f"\n[INFO] Splitting data (by_session={split_by_session})...")
    train_samples, val_samples, test_samples = train_val_test_split(
        samples, train_ratio, val_ratio, split_by_session, seed
    )
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Extract features
    print("\n[INFO] Extracting features...")

    def samples_to_arrays(sample_list: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        features_list = []
        labels_list = []
        for s in sample_list:
            feat = extract_features(s, apply_normalization=False)  # Already normalized
            if feat is not None:
                features_list.append(feat)
                labels_list.append(label_map[s['label']])
        return np.array(features_list), np.array(labels_list)

    train_X, train_y = samples_to_arrays(train_samples)
    val_X, val_y = samples_to_arrays(val_samples)
    test_X, test_y = samples_to_arrays(test_samples)

    print(f"  Train features shape: {train_X.shape}")

    # Compute standardization stats from training set only
    feature_stats = compute_feature_stats(train_X)

    # Create datasets
    train_dataset = HandShapeDataset(train_X, train_y, stats=feature_stats)
    val_dataset = HandShapeDataset(val_X, val_y, stats=feature_stats)
    test_dataset = HandShapeDataset(test_X, test_y, stats=feature_stats)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Save label map and feature stats
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(label_map, out_dir / "label_map.json")
    save_json(feature_stats, out_dir / "feature_stats.json")
    print(f"\n[INFO] Saved label_map.json and feature_stats.json to {out_dir}")

    return train_loader, val_loader, test_loader, label_map, feature_stats, num_classes
