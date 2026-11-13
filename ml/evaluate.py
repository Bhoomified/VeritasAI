"""Evaluation entry point for a trained checkpoint and labelled manifest."""
import argparse
from .dataset_loader import VideoManifestDataset
from .inference import load_checkpoint
from .metrics import classification_metrics
def evaluate(manifest, checkpoint, frames=20):
    dataset = VideoManifestDataset(manifest, frames)
    extractor, classifier, _, device = load_checkpoint(checkpoint)
    labels, probabilities = [], []
    import torch
    with torch.no_grad():
        for images, label, _ in dataset:
            features = extractor(images.to(device)).unsqueeze(0)
            probabilities.append(float(classifier.probability(features).item())); labels.append(int(label.item()))
    return classification_metrics(labels, probabilities)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", required=True); parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(); print(evaluate(args.manifest, args.checkpoint))

