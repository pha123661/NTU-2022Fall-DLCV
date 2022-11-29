import random

from torch import nn


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class Classifier(nn.Module):
    def __init__(self, backbone, in_features, n_class, n_layers=3, dropout=0.1, hidden_size=512) -> None:
        super().__init__()
        layers = []
        layers.extend([
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        ])
        for _ in range(n_layers - 2):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_size, n_class))
        self.backbone = backbone
        self.clf = nn.Sequential(*layers)

    def forward(self, img):
        embeds = self.backbone(img)
        logits = self.clf(embeds)
        return logits
