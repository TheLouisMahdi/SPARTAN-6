"""One small architecture, initialized/trained independently for each encoding."""
import torch
from torch import nn

ARCHITECTURE = {
    "version": "tiny-eye-v1", "input": [1, 16, 32], "classes": {"OPEN": 0, "CLOSED": 1},
    "layers": ["Conv valid 3x3 1->4 + ReLU", "MaxPool 2x2 stride2 floor",
               "Conv valid 3x3 4->8 + ReLU", "MaxPool 2x2 stride2 floor",
               "Flatten C,H,W (96)", "Linear 96->8 + ReLU", "Linear 8->2 logits"],
    "parameters": 1130, "macs_per_eye": 34624,
}


class TinyEye(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.fc1 = nn.Linear(96, 8)
        self.fc2 = nn.Linear(8, 2)

    def activations(self, x):
        a = torch.relu(self.conv1(x))
        b = torch.relu(self.conv2(torch.nn.functional.max_pool2d(a, 2)))
        c = torch.relu(self.fc1(torch.nn.functional.max_pool2d(b, 2).flatten(1)))
        return a, b, c

    def forward(self, x):
        return self.fc2(self.activations(x)[-1])
