"""Per-layer sensitivity analysis and mixed-precision search under an accuracy budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import MLP, accuracy
from .qmodel import QuantConfig, QuantizedMLP


def kl_divergence(reference_logits: np.ndarray, logits: np.ndarray) -> float:
    """Mean KL(p_ref || p) between softmax outputs, a finer signal than accuracy."""

    def log_softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    log_p, log_q = log_softmax(reference_logits.astype(np.float64)), log_softmax(logits.astype(np.float64))
    return float(np.mean(np.sum(np.exp(log_p) * (log_p - log_q), axis=1)))


def layer_sensitivity(
    model: MLP,
    x_calib: np.ndarray,
    bits: int = 4,
    *,
    observer: str = "mse",
    per_channel: bool = True,
) -> list[dict]:
    """Quantize one layer's weights to ``bits`` (everything else FP32) and measure the damage."""
    reference = model.forward(x_calib)
    n = len(model.layers)
    rows = []
    for i in range(n):
        config = QuantConfig.uniform(n, None, None, observer=observer).with_layer(
            i, weight_bits=bits, per_channel=per_channel
        )
        logits = QuantizedMLP(model, config, x_calib).forward(x_calib)
        rows.append(
            {
                "layer": i,
                "shape": list(model.layers[i].weight.shape),
                "bits": bits,
                "kl": kl_divergence(reference, logits),
                "agreement": float((logits.argmax(1) == reference.argmax(1)).mean()),
            }
        )
    return rows


@dataclass
class SearchResult:
    config: QuantConfig
    decisions: list[dict]


def search_mixed_precision(
    model: MLP,
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    *,
    budget: float = 0.005,
    low_bits: int = 4,
    high_bits: int = 8,
    observer: str = "mse",
) -> SearchResult:
    """Greedy bit allocation that stays within ``budget`` accuracy loss on calibration data.

    Start from W``high_bits``A8 everywhere, then try to lower each layer's
    weights to ``low_bits`` in order of increasing sensitivity, keeping a
    change only if calibration accuracy stays within ``budget`` of FP32.
    Largest layers go first among equally sensitive ones, since they save the
    most bytes. The test set is never consulted.
    """
    n = len(model.layers)
    fp32_acc = accuracy(model.forward(x_calib), y_calib)
    sensitivity = layer_sensitivity(model, x_calib, low_bits, observer=observer)
    order = sorted(range(n), key=lambda i: (round(sensitivity[i]["kl"], 6), -model.layers[i].weight.size))
    config = QuantConfig.uniform(n, high_bits, 8, observer=observer)
    decisions = []
    for i in order:
        candidate = config.with_layer(i, weight_bits=low_bits)
        cand_acc = accuracy(QuantizedMLP(model, candidate, x_calib).forward(x_calib), y_calib)
        accepted = fp32_acc - cand_acc <= budget
        decisions.append(
            {
                "layer": i,
                "kl_at_low_bits": sensitivity[i]["kl"],
                "calib_accuracy": cand_acc,
                "accepted": accepted,
            }
        )
        if accepted:
            config = candidate
    bits = "".join(str(s.weight_bits) for s in config.layers)
    return SearchResult(QuantConfig(config.layers, observer, f"Mixed W[{bits}]A8 (searched)"), decisions)
