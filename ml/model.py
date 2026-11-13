from dataclasses import asdict, dataclass
import torch
from torch import nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

@dataclass
class ModelConfig:
    input_size: int = 2048
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    num_frames: int = 20
    model_version: str = "resnext50-lstm-v1"
    backbone_trainable: bool = False

class ResNeXtFeatureExtractor(nn.Module):
    def __init__(self, use_imagenet_weights=False, trainable=False):
        super().__init__()
        weights = ResNeXt50_32X4D_Weights.DEFAULT if use_imagenet_weights else None
        backbone = resnext50_32x4d(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        for parameter in self.features.parameters():
            parameter.requires_grad = trainable
    def forward(self, images):
        return self.features(images).flatten(1)

class LSTMClassifier(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.lstm = nn.LSTM(
            self.config.input_size, self.config.hidden_size, self.config.num_layers,
            batch_first=True, dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            bidirectional=self.config.bidirectional,
        )
        output_size = self.config.hidden_size * (2 if self.config.bidirectional else 1)
        self.classifier = nn.Linear(output_size, 1)
    def forward(self, sequence):
        output, _ = self.lstm(sequence)
        logits = self.classifier(output[:, -1, :]).squeeze(-1)
        return logits
    def probability(self, sequence):
        return torch.sigmoid(self(sequence))

def build_models(config=None, use_imagenet_weights=False):
    config = config or ModelConfig()
    return ResNeXtFeatureExtractor(use_imagenet_weights, config.backbone_trainable), LSTMClassifier(config)

def config_dict(config):
    return asdict(config)

