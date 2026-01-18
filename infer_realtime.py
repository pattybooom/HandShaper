#!/usr/bin/env python3
"""
infer_realtime.py - Real-time inference for HandShaperr hand shape classification
"""

import argparse
import ssl
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from model import HandShapeClassifier
from utils import normalize_landmarks, standardize_features, load_json, PredictionSmoother


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Hand connections for drawing
HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),           # Palm
])

# Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def download_model():
    """Download the hand landmarker model if not present."""
    if not MODEL_PATH.exists():
        print("[INFO] Downloading hand landmarker model...")
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(MODEL_URL, context=ssl_context) as response:
            with open(MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
        print(f"[INFO] Model saved to {MODEL_PATH}")


def draw_hand_landmarks(image: np.ndarray, hand_landmarks) -> None:
    """Draw hand landmarks and connections on the image."""
    h, w = image.shape[:2]

    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            start_pt = (int(start.x * w), int(start.y * h))
            end_pt = (int(end.x * w), int(end.y * h))
            cv2.line(image, start_pt, end_pt, COLOR_GREEN, 2)

    # Draw landmarks
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
        cv2.circle(image, (cx, cy), 5, COLOR_RED, 1)


def extract_features_from_detection(
    results,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, bool, bool]:
    """
    Extract features from MediaPipe detection results.

    Returns:
        Tuple of (features, hand1_detected, hand2_detected).
        Features shape: (128,)
    """
    hand1_landmarks = np.zeros((21, 3), dtype=np.float32)
    hand2_landmarks = np.zeros((21, 3), dtype=np.float32)
    hand1_detected = False
    hand2_detected = False

    if results.hand_landmarks:
        for idx, landmarks in enumerate(results.hand_landmarks):
            lm_array = np.array([
                [lm.x * image_width, lm.y * image_height, lm.z * image_width]
                for lm in landmarks
            ], dtype=np.float32)

            # Normalize
            lm_normalized = normalize_landmarks(lm_array)

            if idx == 0:
                hand1_landmarks = lm_normalized
                hand1_detected = True
            elif idx == 1:
                hand2_landmarks = lm_normalized
                hand2_detected = True

    # Build feature vector
    features = np.concatenate([
        hand1_landmarks.flatten(),  # 63
        hand2_landmarks.flatten(),  # 63
        [float(hand1_detected)],    # 1
        [float(hand2_detected)],    # 1
    ]).astype(np.float32)

    return features, hand1_detected, hand2_detected


# =============================================================================
# REAL-TIME INFERENCE
# =============================================================================


class RealTimeInference:
    """Real-time hand shape classification."""

    def __init__(
        self,
        checkpoint_path: str,
        camera: int = 0,
        min_confidence: float = 0.6,
        smoothing_window: int = 5,
    ):
        self.camera = camera
        self.min_confidence = min_confidence

        # Load checkpoint
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Load label map
        self.label_map = checkpoint['label_map']
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        self.num_classes = len(self.label_map)
        print(f"[INFO] Classes: {list(self.label_map.keys())}")

        # Load feature stats
        self.feature_stats = {
            'mean': np.array(checkpoint['feature_stats']['mean'], dtype=np.float32),
            'std': np.array(checkpoint['feature_stats']['std'], dtype=np.float32),
        }

        # Load model
        config = checkpoint.get('config', {})
        self.model = HandShapeClassifier(
            input_dim=128,
            hidden_dim=config.get('hidden', 256),
            num_classes=self.num_classes,
            dropout=0.0,  # No dropout during inference
        )
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        print(f"[INFO] Model loaded (val_acc: {checkpoint.get('val_acc', 'N/A'):.4f})")

        # Setup MediaPipe
        download_model()
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Prediction smoother
        self.smoother = PredictionSmoother(window_size=smoothing_window)

        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()

    def predict(self, features: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        """
        Make prediction from features.

        Returns:
            Tuple of (top1_label, top1_prob, top3_list).
        """
        # Standardize features
        features_std = standardize_features(features.reshape(1, -1), self.feature_stats)

        # Predict
        with torch.no_grad():
            x = torch.from_numpy(features_std)
            probs = self.model.predict_proba(x)[0].numpy()

        # Apply smoothing
        probs = self.smoother.update(probs)

        # Get top predictions
        top_indices = np.argsort(probs)[::-1]

        top1_idx = top_indices[0]
        top1_label = self.idx_to_label[top1_idx]
        top1_prob = probs[top1_idx]

        top3 = [
            (self.idx_to_label[idx], probs[idx])
            for idx in top_indices[:3]
        ]

        return top1_label, top1_prob, top3

    def update_fps(self) -> float:
        """Update and return FPS."""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt > 0:
            self.fps_history.append(1.0 / dt)

        return np.mean(self.fps_history) if self.fps_history else 0.0

    def draw_ui(
        self,
        frame: np.ndarray,
        prediction: str,
        confidence: float,
        top3: list[tuple[str, float]],
        hands_detected: bool,
        fps: float,
    ) -> np.ndarray:
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background for UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), COLOR_BLACK, -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Title
        cv2.putText(frame, "HandShaperr", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)

        # Prediction
        if hands_detected:
            pred_color = COLOR_GREEN if confidence > 0.7 else COLOR_YELLOW
            cv2.putText(frame, f"Shape: {prediction}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

            # Top-3 predictions
            y_offset = 130
            for i, (label, prob) in enumerate(top3[:3]):
                cv2.putText(frame, f"  {i+1}. {label}: {prob:.1%}", (20, y_offset + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        else:
            cv2.putText(frame, "No hands detected", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        # FPS (bottom left)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

        # Controls (bottom right)
        cv2.putText(frame, "[Q] Quit", (w - 100, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

        return frame

    def run(self):
        """Main inference loop."""
        print("\n" + "=" * 50)
        print("HandShaperr Real-Time Inference")
        print("=" * 50)
        print("Controls: [Q] Quit")
        print("=" * 50 + "\n")

        # Open camera
        cap = cv2.VideoCapture(self.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("[ERROR] Could not open camera")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame")
                    break

                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # Detect hands
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = self.detector.detect(mp_image)

                # Extract features and draw landmarks
                features, h1_det, h2_det = extract_features_from_detection(results, w, h)
                hands_detected = h1_det or h2_det

                # Draw landmarks
                if results.hand_landmarks:
                    for landmarks in results.hand_landmarks:
                        draw_hand_landmarks(frame, landmarks)

                # Predict
                if hands_detected:
                    prediction, confidence, top3 = self.predict(features)
                else:
                    prediction, confidence, top3 = "N/A", 0.0, []
                    self.smoother.reset()

                # Update FPS
                fps = self.update_fps()

                # Draw UI
                frame = self.draw_ui(frame, prediction, confidence, top3, hands_detected, fps)

                # Show frame
                cv2.imshow("HandShaperr - Real-Time Inference", frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            print("\n[INFO] Inference stopped")


# =============================================================================
# MAIN
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time hand shape classification inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best.pt checkpoint file")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index")
    parser.add_argument("--min_confidence", type=float, default=0.6,
                        help="Minimum hand detection confidence")
    parser.add_argument("--smoothing_window", type=int, default=5,
                        help="Window size for prediction smoothing")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    inference = RealTimeInference(
        checkpoint_path=args.checkpoint,
        camera=args.camera,
        min_confidence=args.min_confidence,
        smoothing_window=args.smoothing_window,
    )
    inference.run()
