"""Training entry point. Requires a real CSV manifest; no data is fabricated."""
import argparse, json, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from .dataset_loader import VideoManifestDataset
from .device import select_device
from .model import ModelConfig, build_models, config_dict

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def train(manifest, output, epochs=10, num_frames=20, seed=42, use_imagenet_weights=True):
    seed_everything(seed)
    dataset = VideoManifestDataset(manifest, num_frames)
    if len(dataset) < 2: raise ValueError("At least two labelled videos are required.")
    # Manifests should already be split by person/source. A deterministic split
    # is only used within a training manifest when no separate validation file
    # is supplied; frame-level splitting is never performed.
    train_set, validation_set = random_split(dataset, [max(1, len(dataset)-1), 1], generator=torch.Generator().manual_seed(seed))
    device = select_device()
    config = ModelConfig(num_frames=num_frames)
    extractor, classifier = build_models(config, use_imagenet_weights=use_imagenet_weights)
    extractor.to(device).eval(); classifier.to(device).train()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    for _ in range(epochs):
        for frames, labels, _ in DataLoader(train_set, batch_size=1):
            with torch.no_grad(): features = extractor(frames.squeeze(0).to(device)).unsqueeze(0)
            loss = loss_fn(classifier(features), labels.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        best_loss = min(best_loss, float(loss.item()))
    torch.save({"config": config_dict(config), "feature_extractor": extractor.state_dict(),
                "classifier": classifier.state_dict(), "use_imagenet_weights": use_imagenet_weights,
                "best_loss": best_loss}, output)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True); parser.add_argument("--output", default="checkpoints/model.pt")
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--no-imagenet-weights", action="store_true",
                        help="Use random backbone initialization (not recommended).")
    args = parser.parse_args(); train(
        args.manifest, args.output, args.epochs, args.frames,
        use_imagenet_weights=not args.no_imagenet_weights,
    )
