#!/usr/bin/env python3
"""
utils.py - Utility functions for HandShaperr
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks:
    1. Translate so wrist (landmark 0) is at origin.
    2. Scale by distance from wrist to middle finger MCP (landmark 9).

    Args:
        landmarks: Array of shape (21, 3) with landmark coordinates.

    Returns:
        Normalized landmarks of shape (21, 3).
    """
    WRIST_IDX = 0
    MIDDLE_MCP_IDX = 9

    # Already zeros (missing hand)
    if np.allclose(landmarks, 0):
        return landmarks

    # Translate to wrist origin
    wrist = landmarks[WRIST_IDX].copy()
    translated = landmarks - wrist

    # Scale by wrist-to-middle_mcp distance
    middle_mcp = translated[MIDDLE_MCP_IDX]
    scale = np.linalg.norm(middle_mcp)

    if scale > 1e-6:
        normalized = translated / scale
    else:
        normalized = translated

    return normalized.astype(np.float32)


def compute_feature_stats(features: np.ndarray) -> dict:
    """
    Compute mean and std for feature standardization.

    Args:
        features: Array of shape (N, D) with feature vectors.

    Returns:
        Dict with 'mean' and 'std' arrays.
    """
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    # Prevent division by zero
    std = np.where(std < 1e-6, 1.0, std)
    return {'mean': mean, 'std': std}


def standardize_features(features: np.ndarray, stats: dict) -> np.ndarray:
    """Apply z-score standardization using precomputed stats."""
    return ((features - stats['mean']) / stats['std']).astype(np.float32)


def save_json(data: dict, path: Path) -> None:
    """Save dictionary to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


def load_json(path: Path) -> dict:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def create_run_dir(base_dir: Path, prefix: str = "run") -> Path:
    """Create a timestamped run directory."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class PredictionSmoother:
    """Moving average smoother for prediction probabilities."""

    def __init__(self, window_size: int = 5, num_classes: Optional[int] = None):
        self.window_size = window_size
        self.history: list[np.ndarray] = []
        self.num_classes = num_classes

    def update(self, probs: np.ndarray) -> np.ndarray:
        """
        Add new prediction and return smoothed probabilities.

        Args:
            probs: Probability distribution over classes.

        Returns:
            Smoothed probability distribution.
        """
        self.history.append(probs)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Average over history
        smoothed = np.mean(self.history, axis=0)
        return smoothed

    def reset(self) -> None:
        """Clear prediction history."""
        self.history.clear()
