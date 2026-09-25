"""Per-layer sensitivity analysis and mixed-precision search under an accuracy budget."""

from __future__ import annotations

import itertools
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
    order: str = "kl",
) -> SearchResult:
    """Greedy bit allocation that stays within ``budget`` accuracy loss on calibration data.

    Start from W``high_bits``A8 everywhere, then try to lower each layer's
    weights to ``low_bits`` one at a time, keeping a change only if calibration
    accuracy stays within ``budget`` of FP32. The test set is never consulted.

    ``order="kl"`` tries layers in order of increasing sensitivity (largest
    first on ties). ``order="kl_per_byte"`` divides each layer's sensitivity by
    the bytes it would save, so a large, slightly sensitive layer is tried
    before a tiny, insensitive one; see ``search_ablation`` for why that matters.
    """
    n = len(model.layers)
    fp32_acc = accuracy(model.forward(x_calib), y_calib)
    sensitivity = layer_sensitivity(model, x_calib, low_bits, observer=observer)
    if order == "kl":
        order_key = lambda i: (round(sensitivity[i]["kl"], 6), -model.layers[i].weight.size)  # noqa: E731
    elif order == "kl_per_byte":
        order_key = lambda i: sensitivity[i]["kl"] / model.layers[i].weight.size  # noqa: E731
    else:
        raise ValueError(f"unknown order {order!r}")
    order = sorted(range(n), key=order_key)
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


def exhaustive_search(
    model: MLP,
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    *,
    budget: float = 0.005,
    low_bits: int = 4,
    high_bits: int = 8,
    observer: str = "mse",
) -> SearchResult:
    """Try all 2^n weight bit assignments and keep the smallest one within ``budget``.

    This is the reference answer the greedy search is checked against. It is only
    practical for small models: the cost doubles with every layer, while the greedy
    search needs 2n quantized forward passes. Ties on size go to the lower calibration
    KL. If no assignment meets the budget, it falls back to ``high_bits`` everywhere,
    as the greedy search does.
    """
    n = len(model.layers)
    reference = model.forward(x_calib)
    fp32_acc = accuracy(reference, y_calib)
    base = QuantConfig.uniform(n, high_bits, 8, observer=observer)
    best, decisions = None, []
    for bits in itertools.product((high_bits, low_bits), repeat=n):
        config = base
        for i, b in enumerate(bits):
            config = config.with_layer(i, weight_bits=b)
        qmodel = QuantizedMLP(model, config, x_calib)
        logits = qmodel.forward(x_calib)
        row = {
            "weight_bits": list(bits),
            "calib_accuracy": accuracy(logits, y_calib),
            "kl": kl_divergence(reference, logits),
            "bytes": qmodel.weight_bytes(),
        }
        row["within_budget"] = fp32_acc - row["calib_accuracy"] <= budget
        decisions.append(row)
        if row["within_budget"] and (best is None or (row["bytes"], row["kl"]) < (best[0]["bytes"], best[0]["kl"])):
            best = (row, config)
    config = best[1] if best else base
    bits = "".join(str(s.weight_bits) for s in config.layers)
    return SearchResult(QuantConfig(config.layers, observer, f"Mixed W[{bits}]A8 (exhaustive)"), decisions)
