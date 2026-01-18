#!/usr/bin/env python3
"""
ws_server.py - WebSocket server for HandShaperr real-time predictions

Receives hand landmarks from browser, runs inference, returns predictions.

Usage:
    python ws_server.py --checkpoint ./runs/run1/best.pt --port 8000
"""

import argparse
import asyncio
import json
import ssl
from pathlib import Path

import numpy as np
import torch

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    exit(1)

from model import HandShapeClassifier
from utils import normalize_landmarks, standardize_features


# =============================================================================
# Inference Engine
# =============================================================================


class InferenceEngine:
    """Handles model loading and inference."""

    def __init__(self, checkpoint_path: str):
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
            dropout=0.0,
        )
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        print(f"[INFO] Model loaded (val_acc: {checkpoint.get('val_acc', 'N/A'):.4f})")

    def extract_features(self, hands_data: list) -> np.ndarray:
        """
        Extract 128-dim feature vector from hands data.

        Args:
            hands_data: List of hand dicts with 'handedness' and 'landmarks'.

        Returns:
            Feature vector of shape (128,).
        """
        hand1_landmarks = np.zeros((21, 3), dtype=np.float32)
        hand2_landmarks = np.zeros((21, 3), dtype=np.float32)
        hand1_detected = 0.0
        hand2_detected = 0.0

        for i, hand in enumerate(hands_data[:2]):
            landmarks = hand.get('landmarks', [])
            if len(landmarks) != 21:
                continue

            # Convert to numpy array
            lm_array = np.array([
                [lm['x'], lm['y'], lm['z']]
                for lm in landmarks
            ], dtype=np.float32)

            # Normalize (translate to wrist, scale by hand size)
            lm_normalized = normalize_landmarks(lm_array)

            if i == 0:
                hand1_landmarks = lm_normalized
                hand1_detected = 1.0
            elif i == 1:
                hand2_landmarks = lm_normalized
                hand2_detected = 1.0

        # Build feature vector
        features = np.concatenate([
            hand1_landmarks.flatten(),  # 63
            hand2_landmarks.flatten(),  # 63
            [hand1_detected],           # 1
            [hand2_detected],           # 1
        ]).astype(np.float32)

        return features

    def predict(self, hands_data: list) -> dict:
        """
        Run inference on hands data.

        Returns:
            Dict with 'label', 'prob', 'topk'.
        """
        if not hands_data:
            return {'label': None, 'prob': 0.0, 'topk': []}

        # Extract features
        features = self.extract_features(hands_data)

        # Standardize
        features_std = standardize_features(features.reshape(1, -1), self.feature_stats)

        # Predict
        with torch.no_grad():
            x = torch.from_numpy(features_std)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)[0].numpy()

        # Get top-k
        top_indices = np.argsort(probs)[::-1]
        topk = [
            {'label': self.idx_to_label[idx], 'prob': float(probs[idx])}
            for idx in top_indices[:3]
        ]

        top1_idx = top_indices[0]
        return {
            'label': self.idx_to_label[top1_idx],
            'prob': float(probs[top1_idx]),
            'topk': topk,
        }


# =============================================================================
# WebSocket Server
# =============================================================================


class WebSocketServer:
    """WebSocket server for real-time inference."""

    def __init__(self, engine: InferenceEngine, host: str = '0.0.0.0', port: int = 8000):
        self.engine = engine
        self.host = host
        self.port = port
        self.clients = set()

    async def handler(self, websocket):
        """Handle a single WebSocket connection."""
        client_id = id(websocket)
        self.clients.add(websocket)
        print(f"[WS] Client connected: {client_id} (total: {len(self.clients)})")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)

                    # Extract hands data
                    hands = data.get('hands', [])
                    timestamp = data.get('t', 0)

                    # Run inference
                    result = self.engine.predict(hands)
                    result['t'] = timestamp

                    # Send response
                    await websocket.send(json.dumps(result))

                except json.JSONDecodeError as e:
                    print(f"[WS] JSON parse error: {e}")
                except Exception as e:
                    print(f"[WS] Inference error: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[WS] Client disconnected: {client_id} (total: {len(self.clients)})")

    async def run(self):
        """Start the WebSocket server."""
        print(f"\n{'='*50}")
        print(f"HandShaperr WebSocket Server")
        print(f"{'='*50}")
        print(f"Listening on ws://{self.host}:{self.port}/ws")
        print(f"Press Ctrl+C to stop")
        print(f"{'='*50}\n")

        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
        ):
            await asyncio.Future()  # Run forever


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="WebSocket server for HandShaperr inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=str, default="./runs/run1/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    engine = InferenceEngine(args.checkpoint)

    # Start server
    server = WebSocketServer(engine, args.host, args.port)

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped")


if __name__ == "__main__":
    main()
