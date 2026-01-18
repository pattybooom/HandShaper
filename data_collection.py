#!/usr/bin/env python3
"""
HandShaperr - Data Collection Script for Hand-Formed Shape Classification
==========================================================================

1) INTENDED PURPOSE:
   - This is NOT sign language translation.
   - This is NOT generic gesture recognition like "thumbs up".
   - This is NOT pure pose estimation.
   - It IS "hand-formed shape classification" from hand pose landmarks:
     Classify geometric shapes formed with hands/fingers (square, triangle,
     circle, star, etc.) based on MediaPipe hand landmark coordinates, in real time.

2) TECHNOLOGIES:
   - OpenCV (cv2) for camera capture + preview window + keyboard controls.
   - MediaPipe Hands for hand landmark detection (21 landmarks per hand).
   - Model training later: PyTorch OR TensorFlow (data saved in framework-agnostic format).
   - Camera: built-in laptop webcam (no special hardware assumed).

3) DATA FORMAT COLLECTED:
   - Landmark coordinates (NOT raw images) for robustness and speed.
   - Per frame: 21 landmarks per hand, each with (x, y, z).
   - Includes handedness (Left/Right) and detection confidence.
   - Normalized features:
     - Translated so wrist landmark is origin.
     - Scaled by hand size (wrist->middle_mcp distance).
   - Two hands: both stored; one hand: second filled with zeros + mask.
   - Saved as: CSV, JSONL (one sample per line), and NumPy .npz.
   - Includes: label, timestamp, sample_id, session_id.

4) WORKFLOWS:
   - Set label at runtime, collect fixed samples per label.
   - Support multiple sessions, append to existing dataset.
   - On-screen: label, samples collected, fps, hand detection, recording state.

CONTROLS:
   [L] - Set label (prompt in terminal)
   [R] - Toggle recording on/off
   [C] - Capture single sample (even if recording off)
   [N] - Start new session folder
   [Q] - Quit cleanly and save

USAGE:
   python3 data_collection.py
   python3 data_collection.py --out_dir ./my_dataset --camera 0 --min_confidence 0.7
   python3 data_collection.py --two_hands False --record_every_n 3

DEPENDENCIES:
   pip install opencv-python mediapipe numpy
"""

import argparse
import csv
import json
import os
import ssl
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# MediaPipe Tasks API imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =============================================================================
# CONSTANTS
# =============================================================================

NUM_LANDMARKS = 21
LANDMARK_DIMS = 3  # x, y, z
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

# MediaPipe model
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_ORANGE = (0, 165, 255)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def normalize_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Normalize hand landmarks:
    1. Translate so wrist (landmark 0) is at origin.
    2. Scale by distance from wrist to middle finger MCP (landmark 9).

    Args:
        landmarks: Array of shape (21, 3) with raw landmark coordinates.

    Returns:
        Tuple of (normalized_landmarks, scale_factor).
        normalized_landmarks has shape (21, 3).
    """
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
        scale = 1.0

    return normalized, scale


def landmarks_to_array(hand_landmarks, image_width: int, image_height: int) -> np.ndarray:
    """
    Convert MediaPipe hand landmarks to numpy array.

    Args:
        hand_landmarks: List of NormalizedLandmark from MediaPipe Tasks API.
        image_width: Width of the input image.
        image_height: Height of the input image.

    Returns:
        Array of shape (21, 3) with (x, y, z) coordinates.
        x, y are in pixel coordinates; z is relative depth.
    """
    coords = np.zeros((NUM_LANDMARKS, LANDMARK_DIMS), dtype=np.float32)
    for i, lm in enumerate(hand_landmarks):
        coords[i, 0] = lm.x * image_width
        coords[i, 1] = lm.y * image_height
        coords[i, 2] = lm.z * image_width  # z is relative to x scale
    return coords


def create_empty_hand_data() -> dict:
    """Create empty hand data structure for missing hand."""
    return {
        "landmarks_raw": np.zeros((NUM_LANDMARKS, LANDMARK_DIMS), dtype=np.float32),
        "landmarks_normalized": np.zeros((NUM_LANDMARKS, LANDMARK_DIMS), dtype=np.float32),
        "handedness": "None",
        "confidence": 0.0,
        "scale": 0.0,
        "detected": False,
    }


def process_hand(hand_landmarks, handedness_info, image_width: int, image_height: int) -> dict:
    """
    Process a single detected hand.

    Returns:
        Dictionary with raw landmarks, normalized landmarks, handedness, confidence, etc.
    """
    raw_landmarks = landmarks_to_array(hand_landmarks, image_width, image_height)
    normalized_landmarks, scale = normalize_landmarks(raw_landmarks)

    # New Tasks API: handedness_info is a Category object
    return {
        "landmarks_raw": raw_landmarks,
        "landmarks_normalized": normalized_landmarks,
        "handedness": handedness_info.category_name,
        "confidence": handedness_info.score,
        "scale": scale,
        "detected": True,
    }


def download_model():
    """Download the hand landmarker model if not present."""
    if not MODEL_PATH.exists():
        print(f"[INFO] Downloading hand landmarker model...")
        # Create SSL context that doesn't verify certificates (for macOS compatibility)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(MODEL_URL, context=ssl_context) as response:
            with open(MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
        print(f"[INFO] Model saved to {MODEL_PATH}")


# MediaPipe hand connections (21 landmarks)
# Defined manually to avoid dependency on solutions module
HAND_CONNECTIONS = frozenset([
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
])


def draw_hand_landmarks(image: np.ndarray, hand_landmarks, connections=None):
    """Draw hand landmarks on the image."""
    h, w = image.shape[:2]

    # Default MediaPipe hand connections
    if connections is None:
        connections = HAND_CONNECTIONS

    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    # Draw landmarks
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), 1)


# =============================================================================
# DATA STORAGE CLASS
# =============================================================================


class DatasetManager:
    """Manages dataset storage, sessions, and saving."""

    def __init__(self, out_dir: str, session_id: Optional[str] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.out_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.samples: list[dict] = []
        self.sample_counter = 0
        self.class_counts: dict[str, int] = {}

        # File paths
        self.csv_path = self.session_dir / "samples.csv"
        self.jsonl_path = self.session_dir / "samples.jsonl"
        self.npz_path = self.session_dir / "samples.npz"
        self.metadata_path = self.session_dir / "metadata.json"

        # Load existing data if present
        self._load_existing()

        # CSV writer setup
        self._setup_csv()

    def _load_existing(self):
        """Load existing session data if available."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
                self.sample_counter = metadata.get("total_samples", 0)
                self.class_counts = metadata.get("class_counts", {})
            print(f"[INFO] Loaded existing session with {self.sample_counter} samples")

    def _setup_csv(self):
        """Setup CSV file with headers if new."""
        self.csv_file = open(self.csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # Write header if file is new/empty
        if self.csv_path.stat().st_size == 0:
            headers = self._get_csv_headers()
            self.csv_writer.writerow(headers)
            self.csv_file.flush()

    def _get_csv_headers(self) -> list[str]:
        """Generate CSV column headers."""
        headers = ["sample_id", "session_id", "timestamp", "label"]

        # Hand 1 landmarks
        for i in range(NUM_LANDMARKS):
            for dim in ["x", "y", "z"]:
                headers.append(f"h1_lm{i}_{dim}")
        headers.extend(["h1_handedness", "h1_confidence", "h1_scale", "h1_detected"])

        # Hand 2 landmarks
        for i in range(NUM_LANDMARKS):
            for dim in ["x", "y", "z"]:
                headers.append(f"h2_lm{i}_{dim}")
        headers.extend(["h2_handedness", "h2_confidence", "h2_scale", "h2_detected"])

        return headers

    def add_sample(
        self,
        label: str,
        hand1_data: dict,
        hand2_data: dict,
        max_per_label: Optional[int] = None,
    ) -> bool:
        """
        Add a new sample to the dataset.

        Returns:
            True if sample was added, False if max reached for this label.
        """
        # Check if max reached for this label
        if max_per_label is not None:
            if self.class_counts.get(label, 0) >= max_per_label:
                return False

        timestamp = datetime.now().isoformat()
        sample_id = f"{self.session_id}_{self.sample_counter:06d}"

        sample = {
            "sample_id": sample_id,
            "session_id": self.session_id,
            "timestamp": timestamp,
            "label": label,
            "hand1": hand1_data,
            "hand2": hand2_data,
        }

        self.samples.append(sample)
        self.sample_counter += 1
        self.class_counts[label] = self.class_counts.get(label, 0) + 1

        # Save incrementally
        self._save_sample_csv(sample)
        self._save_sample_jsonl(sample)

        return True

    def _save_sample_csv(self, sample: dict):
        """Append a single sample to CSV."""
        row = [
            sample["sample_id"],
            sample["session_id"],
            sample["timestamp"],
            sample["label"],
        ]

        # Hand 1 landmarks (flattened)
        row.extend(sample["hand1"]["landmarks_normalized"].flatten().tolist())
        row.extend([
            sample["hand1"]["handedness"],
            sample["hand1"]["confidence"],
            sample["hand1"]["scale"],
            int(sample["hand1"]["detected"]),
        ])

        # Hand 2 landmarks (flattened)
        row.extend(sample["hand2"]["landmarks_normalized"].flatten().tolist())
        row.extend([
            sample["hand2"]["handedness"],
            sample["hand2"]["confidence"],
            sample["hand2"]["scale"],
            int(sample["hand2"]["detected"]),
        ])

        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def _save_sample_jsonl(self, sample: dict):
        """Append a single sample to JSONL."""
        # Convert numpy arrays to lists for JSON serialization
        json_sample = {
            "sample_id": sample["sample_id"],
            "session_id": sample["session_id"],
            "timestamp": sample["timestamp"],
            "label": sample["label"],
            "hand1": {
                "landmarks_normalized": sample["hand1"]["landmarks_normalized"].tolist(),
                "handedness": sample["hand1"]["handedness"],
                "confidence": float(sample["hand1"]["confidence"]),
                "scale": float(sample["hand1"]["scale"]),
                "detected": sample["hand1"]["detected"],
            },
            "hand2": {
                "landmarks_normalized": sample["hand2"]["landmarks_normalized"].tolist(),
                "handedness": sample["hand2"]["handedness"],
                "confidence": float(sample["hand2"]["confidence"]),
                "scale": float(sample["hand2"]["scale"]),
                "detected": sample["hand2"]["detected"],
            },
        }

        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(json_sample) + "\n")

    def save_npz(self):
        """Save all samples to NPZ format."""
        if not self.samples:
            return

        # Prepare arrays
        n_samples = len(self.samples)
        hand1_landmarks = np.zeros((n_samples, NUM_LANDMARKS, LANDMARK_DIMS), dtype=np.float32)
        hand2_landmarks = np.zeros((n_samples, NUM_LANDMARKS, LANDMARK_DIMS), dtype=np.float32)
        hand1_detected = np.zeros(n_samples, dtype=bool)
        hand2_detected = np.zeros(n_samples, dtype=bool)
        labels = []
        sample_ids = []

        for i, sample in enumerate(self.samples):
            hand1_landmarks[i] = sample["hand1"]["landmarks_normalized"]
            hand2_landmarks[i] = sample["hand2"]["landmarks_normalized"]
            hand1_detected[i] = sample["hand1"]["detected"]
            hand2_detected[i] = sample["hand2"]["detected"]
            labels.append(sample["label"])
            sample_ids.append(sample["sample_id"])

        np.savez(
            self.npz_path,
            hand1_landmarks=hand1_landmarks,
            hand2_landmarks=hand2_landmarks,
            hand1_detected=hand1_detected,
            hand2_detected=hand2_detected,
            labels=np.array(labels),
            sample_ids=np.array(sample_ids),
        )

    def save_metadata(self):
        """Save session metadata."""
        metadata = {
            "session_id": self.session_id,
            "created": datetime.now().isoformat(),
            "total_samples": self.sample_counter,
            "class_counts": self.class_counts,
            "num_landmarks": NUM_LANDMARKS,
            "landmark_dims": LANDMARK_DIMS,
        }

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def close(self):
        """Clean up and save final data."""
        self.csv_file.close()
        self.save_npz()
        self.save_metadata()

    def print_summary(self):
        """Print dataset summary."""
        print("\n" + "=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)
        print(f"Session ID: {self.session_id}")
        print(f"Output directory: {self.session_dir}")
        print(f"Total samples: {self.sample_counter}")
        print("\nSamples per class:")
        for label, count in sorted(self.class_counts.items()):
            print(f"  {label}: {count}")
        print("=" * 50)


# =============================================================================
# MAIN COLLECTOR CLASS
# =============================================================================


class HandShapeCollector:
    """Main data collection class."""

    def __init__(self, args):
        self.args = args

        # Download model if needed
        download_model()

        # MediaPipe Tasks API setup
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2 if args.two_hands else 1,
            min_hand_detection_confidence=args.min_confidence,
            min_hand_presence_confidence=args.min_confidence,
            min_tracking_confidence=args.min_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Camera setup
        self.cap = cv2.VideoCapture(args.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Dataset manager
        self.dataset = DatasetManager(args.out_dir)

        # State
        self.current_label = "unlabeled"
        self.is_recording = False
        self.frame_counter = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0

    def start_new_session(self):
        """Start a new recording session."""
        self.dataset.close()
        self.dataset.print_summary()
        self.dataset = DatasetManager(self.args.out_dir)
        print(f"\n[INFO] Started new session: {self.dataset.session_id}")

    def set_label(self):
        """Prompt user to set current label."""
        print("\n" + "-" * 40)
        new_label = input("Enter shape label (e.g., square, triangle, circle): ").strip().lower()
        if new_label:
            self.current_label = new_label
            print(f"[INFO] Label set to: {self.current_label}")
        else:
            print("[INFO] Label unchanged")
        print("-" * 40)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict, dict, float]:
        """
        Process a single frame with MediaPipe.

        Returns:
            Tuple of (annotated_frame, hand1_data, hand2_data, max_confidence).
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        results = self.detector.detect(mp_image)

        h, w = frame.shape[:2]
        annotated = frame.copy()

        hand1_data = create_empty_hand_data()
        hand2_data = create_empty_hand_data()
        max_confidence = 0.0

        if results.hand_landmarks and results.handedness:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.hand_landmarks, results.handedness)
            ):
                # Draw landmarks
                draw_hand_landmarks(annotated, hand_landmarks)

                # Process hand data (handedness[0] is the primary category)
                hand_data = process_hand(hand_landmarks, handedness[0], w, h)
                max_confidence = max(max_confidence, hand_data["confidence"])

                # Assign to hand1 or hand2 (by detection order)
                if idx == 0:
                    hand1_data = hand_data
                elif idx == 1:
                    hand2_data = hand_data

        return annotated, hand1_data, hand2_data, max_confidence

    def draw_ui(
        self,
        frame: np.ndarray,
        hand1_detected: bool,
        hand2_detected: bool,
        confidence: float,
    ) -> np.ndarray:
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]

        # Semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), COLOR_BLACK, -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Recording indicator
        rec_color = COLOR_RED if self.is_recording else COLOR_WHITE
        rec_text = "● REC" if self.is_recording else "○ PAUSED"
        cv2.putText(frame, rec_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2)

        # Current label
        cv2.putText(
            frame,
            f"Label: {self.current_label}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_YELLOW,
            2,
        )

        # Sample count for current label
        count = self.dataset.class_counts.get(self.current_label, 0)
        max_text = f"/{self.args.max_per_label}" if self.args.max_per_label else ""
        cv2.putText(
            frame,
            f"Count: {count}{max_text}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_WHITE,
            2,
        )

        # Hand detection status
        hand_status = []
        if hand1_detected:
            hand_status.append("H1")
        if hand2_detected:
            hand_status.append("H2")
        hand_text = f"Hands: {'+'.join(hand_status) if hand_status else 'None'}"
        hand_color = COLOR_GREEN if hand_status else COLOR_RED
        cv2.putText(frame, hand_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

        # Confidence and FPS
        cv2.putText(
            frame,
            f"Conf: {confidence:.2f} | FPS: {self.fps:.1f}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_WHITE,
            1,
        )

        # Controls help (bottom of screen)
        controls = "[L]abel [R]ecord [C]apture [N]ew Session [Q]uit"
        cv2.putText(
            frame,
            controls,
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_WHITE,
            1,
        )

        # Recording border flash
        if self.is_recording and (self.frame_counter // 10) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_RED, 4)

        return frame

    def update_fps(self):
        """Update FPS calculation."""
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time

        if elapsed >= 1.0:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.last_fps_time = current_time

    def should_record_frame(self) -> bool:
        """Check if current frame should be recorded based on record_every_n."""
        return self.frame_counter % self.args.record_every_n == 0

    def capture_sample(self, hand1_data: dict, hand2_data: dict, confidence: float) -> bool:
        """
        Attempt to capture a sample.

        Returns:
            True if sample was captured successfully.
        """
        # Check minimum requirements
        if not hand1_data["detected"]:
            return False

        if confidence < self.args.min_confidence:
            return False

        # Check two_hands requirement
        if self.args.two_hands and not hand2_data["detected"]:
            return False

        # Add sample
        added = self.dataset.add_sample(
            self.current_label,
            hand1_data,
            hand2_data,
            self.args.max_per_label,
        )

        return added

    def run(self):
        """Main collection loop."""
        print("\n" + "=" * 50)
        print("HandShaperr Data Collection")
        print("=" * 50)
        print(f"Output: {self.dataset.session_dir}")
        print(f"Camera: {self.args.camera}")
        print(f"Min confidence: {self.args.min_confidence}")
        print(f"Two hands required: {self.args.two_hands}")
        print(f"Record every N frames: {self.args.record_every_n}")
        print("\nControls:")
        print("  [L] Set label    [R] Toggle recording")
        print("  [C] Capture one  [N] New session")
        print("  [Q] Quit")
        print("=" * 50 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Failed to read from camera")
                    break

                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Process frame
                annotated, hand1_data, hand2_data, confidence = self.process_frame(frame)

                # Handle recording
                if self.is_recording and self.should_record_frame():
                    self.capture_sample(hand1_data, hand2_data, confidence)

                # Draw UI
                display = self.draw_ui(
                    annotated,
                    hand1_data["detected"],
                    hand2_data["detected"],
                    confidence,
                )

                # Show frame
                cv2.imshow("HandShaperr - Data Collection", display)

                # Update counters
                self.frame_counter += 1
                self.update_fps()

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("l"):
                    self.set_label()
                elif key == ord("r"):
                    self.is_recording = not self.is_recording
                    status = "STARTED" if self.is_recording else "STOPPED"
                    print(f"[INFO] Recording {status}")
                elif key == ord("c"):
                    # Capture single sample
                    if self.capture_sample(hand1_data, hand2_data, confidence):
                        print(f"[INFO] Captured sample for '{self.current_label}'")
                    else:
                        print("[WARN] Could not capture - check hand detection/confidence")
                elif key == ord("n"):
                    self.start_new_session()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")

        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            self.dataset.close()
            self.dataset.print_summary()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HandShaperr - Data Collection for Hand-Formed Shape Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_collection.py
  python data_collection.py --out_dir ./my_dataset --camera 0
  python data_collection.py --min_confidence 0.7 --two_hands False
  python data_collection.py --record_every_n 3 --max_per_label 1000
        """,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./dataset",
        help="Output directory for dataset (default: ./dataset)",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )

    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.6,
        help="Minimum detection confidence threshold (default: 0.6)",
    )

    parser.add_argument(
        "--two_hands",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Require two hands for recording (default: True)",
    )

    parser.add_argument(
        "--record_every_n",
        type=int,
        default=2,
        help="Record every N frames to reduce redundancy (default: 2)",
    )

    parser.add_argument(
        "--max_per_label",
        type=int,
        default=None,
        help="Maximum samples per label, optional (default: None = unlimited)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    collector = HandShapeCollector(args)
    collector.run()


if __name__ == "__main__":
    main()