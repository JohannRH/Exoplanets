"""
Step 3 (model definition): PyTorch MLP for exoplanet disposition classification.

Context:
  We need a model that takes a vector of features (e.g. 8 floats: period, duration,
  depth, ...) and outputs scores (logits) for 3 classes. A small fully connected
  network (MLP) is sufficient for this tabular task; we do not use convolutions
  or attention. The loss (CrossEntropyLoss in train.py) will convert logits to
  probabilities and compare to integer labels 0, 1, 2.

What this module provides:
  - ExoplanetMLP: input -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> logits.
  - get_model(): convenience to build the MLP with the right input_size and num_classes.

Important details:
  - Logits are raw scores; no softmax here. PyTorch's CrossEntropyLoss includes
    log_softmax internally, so we pass logits directly.
  - Dropout is only applied during training (model.train()); in eval mode it is off.
  - input_size must match the number of features in the preprocessed data (e.g. 8).
"""

import torch
import torch.nn as nn


class ExoplanetMLP(nn.Module):
    """
    Multi-layer perceptron for 3-class classification.

    Architecture:
        Input (batch_size, input_size)
          -> Linear(input_size, h1) -> ReLU -> Dropout
          -> Linear(h1, h2) -> ReLU -> Dropout
          -> Linear(h2, num_classes)
        Output: logits (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        hidden_sizes: tuple = (64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes

        # Build backbone: sequence of Linear -> ReLU -> Dropout for each hidden size.
        # prev tracks output size of previous layer (start with input_size).
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_size). Returns logits (batch_size, num_classes).
        No softmax — use with CrossEntropyLoss or apply F.softmax for probabilities.
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def get_model(input_size: int, num_classes: int = 3, **kwargs) -> ExoplanetMLP:
    """
    Build the MLP. input_size = number of features (e.g. 8); num_classes = 3.
    Extra kwargs (e.g. hidden_sizes, dropout) are passed to ExoplanetMLP.
    """
    return ExoplanetMLP(input_size=input_size, num_classes=num_classes, **kwargs)
