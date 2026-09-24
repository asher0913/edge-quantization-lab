"""A small NumPy MLP, its training loop and the dataset it is trained on."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dense:
    weight: np.ndarray  # (out_features, in_features); rows are output channels
    bias: np.ndarray  # (out_features,)

    @property
    def parameters(self) -> int:
        return self.weight.size + self.bias.size


@dataclass
class MLP:
    layers: list[Dense]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.forward_trace(x)[-1]

    def forward_trace(self, x: np.ndarray) -> list[np.ndarray]:
        """Return the input to every layer followed by the output logits."""
        trace = [x]
        for i, layer in enumerate(self.layers):
            x = x @ layer.weight.T + layer.bias
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
            trace.append(x)
        return trace

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).argmax(axis=1)

    @property
    def parameters(self) -> int:
        return sum(layer.parameters for layer in self.layers)

    def save(self, path: str | Path) -> None:
        arrays = {}
        for i, layer in enumerate(self.layers):
            arrays[f"w{i}"], arrays[f"b{i}"] = layer.weight, layer.bias
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> MLP:
        data = np.load(path)
        count = len([k for k in data.files if k.startswith("w")])
        return cls([Dense(data[f"w{i}"], data[f"b{i}"]) for i in range(count)])


@dataclass(frozen=True)
class Splits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_calib: np.ndarray
    y_calib: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_digits_splits(seed: int = 0, calibration: int = 256) -> Splits:
    """8×8 handwritten digits (1,797 images, bundled with scikit-learn).

    Pixels are scaled to [0, 1]. The data is split once into train, a small
    calibration set used for *every* quantization decision, and a test set
    that is only used for reporting.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    x_rest, x_test, y_rest, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=seed)
    x_train, x_calib, y_train, y_calib = train_test_split(
        x_rest, y_rest, test_size=calibration, stratify=y_rest, random_state=seed
    )
    return Splits(x_train, y_train, x_calib, y_calib, x_test, y_test)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden: tuple[int, ...] = (256, 128),
    *,
    epochs: int = 80,
    lr: float = 2e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> MLP:
    """Adam + cross-entropy, He initialisation, deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    classes = int(y.max()) + 1
    sizes = [x.shape[1], *hidden, classes]
    layers = [
        Dense(
            (rng.standard_normal((out, inp)) * np.sqrt(2.0 / inp)).astype(np.float32),
            np.zeros(out, dtype=np.float32),
        )
        for inp, out in zip(sizes[:-1], sizes[1:], strict=True)
    ]
    model = MLP(layers)
    params = [p for layer in layers for p in (layer.weight, layer.bias)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0

    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            trace = model.forward_trace(x[idx])
            probs = _softmax(trace[-1])
            grad = probs
            grad[np.arange(len(idx)), y[idx]] -= 1.0
            grad /= len(idx)
            grads = []
            for i in range(len(layers) - 1, -1, -1):
                inputs = trace[i]
                grads.append(grad.sum(axis=0))  # bias
                grads.append(grad.T @ inputs + weight_decay * layers[i].weight)  # weight
                if i > 0:
                    grad = (grad @ layers[i].weight) * (trace[i] > 0)
            grads = grads[::-1]
            step += 1
            for j, (p, g) in enumerate(zip(params, grads, strict=True)):
                m[j] = beta1 * m[j] + (1 - beta1) * g
                v[j] = beta2 * v[j] + (1 - beta2) * g * g
                m_hat = m[j] / (1 - beta1**step)
                v_hat = v[j] / (1 - beta2**step)
                p -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(p.dtype)
    return model


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())
