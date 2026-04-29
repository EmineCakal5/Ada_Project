"""
Threat MLP — PyTorch tabanlı tehdit sınıflandırıcı.
Feature vektörü → LOW / MEDIUM / HIGH / CRITICAL

Sentetik veri üretimi + model eğitimi + inference içerir.
Implementation Plan v2: "Özgün katkı güçlendirmesi"
"""

import os
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Tehdit seviyeleri
LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ─── Model Mimarisi ───────────────────────────────────────────


class ThreatMLP(nn.Module):
    """
    2-3 katmanlı MLP sınıflandırıcı.
    Input: 10 boyut feature vektörü
    Output: 4 sınıf (LOW/MED/HIGH/CRITICAL)

    Mimari: 10 → 32 → 16 → 4
    """

    def __init__(self, input_dim: int = 8, hidden_dims: list = None, output_dim: int = 4):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),   # LayerNorm: batch_size=1 ile de calısır
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ─── Sentetik Veri Üretici ────────────────────────────────────


def generate_synthetic_data(n_samples: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kural motoru mantığıyla tutarlı sentetik eğitim verisi üret.

    Feature vektörü (10 boyut):
        [0] zone_violation_score
        [1] dwell_time_normalized
        [2] velocity_magnitude
        [3] trajectory_variance
        [4] loitering_score
        [5] abandoned_object_score
        [6] time_of_day_sin
        [7] class_risk
        [8] reconnaissance_score
        [9] coordinated_movement_score

    Label: 0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL
    """
    rng = np.random.default_rng(seed)
    X = []
    y = []

    per_class = n_samples // 4

    # LOW — normal davranış
    for _ in range(per_class):
        x = rng.uniform(0, 0.3, 10)
        x[0] = rng.uniform(0, 0.1)    # zone: safe
        x[4] = rng.uniform(0, 0.2)    # loitering: none
        x[5] = rng.uniform(0, 0.1)    # abandoned: none
        x[8] = rng.uniform(0, 0.1)    # reconnaissance: none
        x[9] = rng.uniform(0, 0.1)    # coordinated: none
        X.append(x); y.append(0)

    # MEDIUM — şüpheli ama düşük risk
    for _ in range(per_class):
        x = rng.uniform(0, 0.5, 10)
        x[0] = rng.uniform(0.1, 0.4)
        x[4] = rng.uniform(0.2, 0.5)
        x[5] = rng.uniform(0.1, 0.4)
        x[8] = rng.uniform(0.2, 0.5)
        x[9] = rng.uniform(0.1, 0.4)
        X.append(x); y.append(1)

    # HIGH — tehlikeli davranış
    for _ in range(per_class):
        x = rng.uniform(0.3, 0.8, 10)
        x[0] = rng.uniform(0.5, 1.0)   # zone violation
        x[4] = rng.uniform(0.5, 0.9)   # loitering
        x[5] = rng.uniform(0.4, 0.8)   # abandoned
        x[8] = rng.uniform(0.5, 0.9)   # reconnaissance
        x[9] = rng.uniform(0.4, 0.8)   # coordinated
        X.append(x); y.append(2)

    # CRITICAL — çok yüksek tehdit
    for _ in range(per_class):
        x = rng.uniform(0.6, 1.0, 10)
        x[0] = rng.uniform(0.8, 1.0)
        x[4] = rng.uniform(0.8, 1.0)
        x[5] = rng.uniform(0.7, 1.0)
        x[8] = rng.uniform(0.7, 1.0)
        x[9] = rng.uniform(0.6, 1.0)
        X.append(x); y.append(3)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ─── Trainer ──────────────────────────────────────────────────


class ThreatMLPTrainer:
    """
    Sentetik veri üretip MLP'yi eğitir ve kaydeder.

    Kullanım:
        trainer = ThreatMLPTrainer(config)
        model = trainer.train_and_save()
    """

    def __init__(self, config: dict):
        mlp_cfg = config["mlp"]
        self.input_dim    = mlp_cfg.get("input_dim", 8)
        self.hidden_dims  = mlp_cfg.get("hidden_dims", [32, 16])
        self.output_dim   = mlp_cfg.get("output_dim", 4)
        self.model_path   = mlp_cfg.get("model_path", "models/weights/threat_mlp.pt")
        self.n_samples    = mlp_cfg.get("threshold_train", 500)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def train_and_save(self, epochs: int = 100, lr: float = 0.001) -> ThreatMLP:
        """Sentetik veri üret, eğit ve kaydet."""
        logger.info(f"MLP eğitimi başlıyor: {self.n_samples} sentetik örnek")

        X, y = generate_synthetic_data(self.n_samples)

        # Train/val split
        split = int(0.8 * len(X))
        X_train, X_val = torch.tensor(X[:split]), torch.tensor(X[split:])
        y_train, y_val = torch.tensor(y[:split]), torch.tensor(y[split:])

        dataset = TensorDataset(X_train, y_train)
        loader  = DataLoader(dataset, batch_size=32, shuffle=True)

        model = ThreatMLP(self.input_dim, self.hidden_dims, self.output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        best_val_acc = 0.0
        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val).argmax(dim=1)
                    val_acc = (val_preds == y_val).float().mean().item()
                logger.info(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.3f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), self.model_path)

        logger.info(f"MLP kaydedildi → {self.model_path} (best acc: {best_val_acc:.3f})")
        return model


# ─── Inference ────────────────────────────────────────────────


class ThreatMLPClassifier:
    """
    Kaydedilmiş MLP ile inference yapar.
    Kural motoruna ML katmanı sağlar.

    Kullanım:
        clf = ThreatMLPClassifier(config)
        label, proba = clf.predict(feature_vector)
    """

    def __init__(self, config: dict):
        mlp_cfg = config["mlp"]
        self.input_dim   = mlp_cfg.get("input_dim", 8)
        self.hidden_dims = mlp_cfg.get("hidden_dims", [32, 16])
        self.output_dim  = mlp_cfg.get("output_dim", 4)
        self.model_path  = mlp_cfg.get("model_path", "models/weights/threat_mlp.pt")

        self.model: Optional[ThreatMLP] = None
        self._load_or_train(config)

    def _load_or_train(self, config: dict):
        """Model varsa yükle, yoksa eğit."""
        if os.path.exists(self.model_path):
            try:
                self.model = ThreatMLP(self.input_dim, self.hidden_dims, self.output_dim)
                self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
                self.model.eval()
                logger.info(f"MLP modeli yüklendi: {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Model yüklenemedi, yeniden eğitiliyor: {e}")

        trainer = ThreatMLPTrainer(config)
        self.model = trainer.train_and_save()
        self.model.eval()

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Feature vektörüne göre tehdit seviyesi tahmin et.

        Args:
            feature_vector: (8,) float32 numpy array

        Returns:
            (threat_label, probabilities) — label: LOW/MEDIUM/HIGH/CRITICAL
        """
        if self.model is None:
            return "LOW", np.array([1.0, 0.0, 0.0, 0.0])

        x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1).squeeze().numpy()
            pred   = int(probs.argmax())

        return LABELS[pred], probs
