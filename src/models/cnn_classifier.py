"""
CNN classifier for bearing fault diagnosis using spectrogram images.
Uses a pretrained ResNet-18 with a modified final layer for 10 fault classes.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=10, pretrained=True):
    """Create a ResNet-18 model fine-tuned for bearing fault classification."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


class BearingFaultCNN(nn.Module):
    """Wrapper that exposes a predict method for integration with the pipeline."""

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.model = get_model(num_classes, pretrained)
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def predict(self, x):
        """Return predicted class index and softmax probabilities."""
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        return pred, probs
