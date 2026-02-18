from __future__ import annotations

from torch import Tensor, nn


class SimpleCNN(nn.Module):
    """A compact CNN suitable for CPU training on cats-vs-dogs."""

    def __init__(self, num_classes: int = 2, input_channels: int = 3) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        if input_channels <= 0:
            raise ValueError(f"input_channels must be > 0, got {input_channels}")

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected inputs with shape [N, C, H, W], got ndim={inputs.ndim}")
        return self.classifier(self.features(inputs))
