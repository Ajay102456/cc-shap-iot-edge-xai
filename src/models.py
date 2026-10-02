"""Edge-representative model definitions: a LightGBM classifier (TreeSHAP-
exact-compatible) and a deliberately small MLP (<=50k params) standing in for
the compact DNNs actually deployable on Pi/Jetson-class hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_lightgbm(X_train, y_train, n_classes: int, random_state: int = 42) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=150,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.1,
        objective="multiclass" if n_classes > 2 else "binary",
        num_class=n_classes if n_classes > 2 else None,
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model


class CompactMLP(nn.Module):
    """~<=50k parameters for typical IoT flow feature counts (20-50 features)."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class TrainedMLP:
    model: CompactMLP
    n_features: int
    n_classes: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            return logits.argmax(dim=1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            return F.softmax(logits, dim=1).numpy()


def train_mlp(
    X_train,
    y_train,
    n_features: int,
    n_classes: int,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    random_state: int = 42,
) -> TrainedMLP:
    torch.manual_seed(random_state)
    model = CompactMLP(n_features, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    n = X_t.shape[0]

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            out = model(X_t[idx])
            loss = F.cross_entropy(out, y_t[idx])
            loss.backward()
            opt.step()

    return TrainedMLP(model=model, n_features=n_features, n_classes=n_classes)
