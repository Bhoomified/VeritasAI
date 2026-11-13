#!/usr/bin/env python
"""Diagnostic-only VeritasAI ML pipeline check.

This script does not load a trained checkpoint and does not emit a REAL/FAKE
prediction. It validates preprocessing, ImageNet ResNeXt50 features, and the
LSTM forward pass using one real video.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

# Allow direct execution from the repository root without requiring PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.device import select_device
from ml.model import LSTMClassifier, ModelConfig, ResNeXtFeatureExtractor
from ml.preprocess import PreprocessingError, extract_frames


def fail(message):
    print(f"ERROR: {message}")
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()

    if not args.video.is_file():
        return fail(f"video does not exist: {args.video}")
    if args.frames <= 0:
        return fail("--frames must be greater than zero")

    print("VeritasAI ML pipeline diagnostic")
    print(f"video: {args.video}")
    print(f"requested frames: {args.frames}")
    device = select_device(args.device)
    print(f"device: {device}")

    started = time.perf_counter()
    try:
        sampled = extract_frames(str(args.video), args.frames)
    except PreprocessingError as exc:
        return fail(f"preprocessing failed: {exc}")
    print(f"decoded frames: {len(sampled.indices)}")
    print(f"frame indices: {sampled.indices}")
    print(f"preprocessed tensor: {tuple(sampled.frames.shape)}")
    print(f"faces detected: {sampled.faces_found}/{len(sampled.indices)}")
    if sampled.frames.ndim != 4 or tuple(sampled.frames.shape[1:]) != (3, 224, 224):
        return fail("expected preprocessed frames with shape [N, 3, 224, 224]")

    print("loading ResNeXt50 ImageNet weights...")
    try:
        feature_extractor = ResNeXtFeatureExtractor(use_imagenet_weights=True)
        feature_extractor.to(device).eval()
    except Exception as exc:
        return fail(f"ResNeXt50/ImageNet initialization failed: {exc}")

    try:
        with torch.no_grad():
            features = feature_extractor(sampled.frames.to(device))
    except Exception as exc:
        return fail(f"feature extraction failed: {exc}")
    print(f"feature tensor: {tuple(features.shape)}")
    expected_features = (len(sampled.indices), 2048)
    if tuple(features.shape) != expected_features:
        return fail(f"expected feature tensor {expected_features}, got {tuple(features.shape)}")

    config = ModelConfig(num_frames=len(sampled.indices))
    classifier = LSTMClassifier(config).to(device).eval()
    sequence = features.unsqueeze(0)
    print(f"LSTM input tensor: {tuple(sequence.shape)}")
    try:
        with torch.no_grad():
            logits = classifier(sequence)
    except Exception as exc:
        return fail(f"LSTM forward pass failed: {exc}")
    print(f"LSTM logits tensor: {tuple(logits.shape)}")
    if tuple(logits.shape) != (1,):
        return fail(f"expected LSTM output shape (1,), got {tuple(logits.shape)}")

    duration = time.perf_counter() - started
    print(f"pipeline elapsed seconds: {duration:.2f}")
    print("STATUS: preprocessing, feature extraction, and LSTM forward pass succeeded")
    print("MODEL STATUS: no trained checkpoint loaded; no REAL/FAKE prediction made")
    return 0


if __name__ == "__main__":
    sys.exit(main())
