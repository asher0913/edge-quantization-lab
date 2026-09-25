"""Multi-seed benchmark: accuracy, fidelity and size for each quantization recipe."""

from __future__ import annotations

import statistics
from collections.abc import Callable

import numpy as np

from .model import MLP, Splits, accuracy, load_digits_splits, train_mlp
from .qmodel import IntegerMLP, QuantConfig, QuantizedMLP, fake_quant_codes
from .search import exhaustive_search, kl_divergence, search_mixed_precision

N_LAYERS = 3


def _uniform(w, a, per_channel=True, observer="minmax", name=""):
    return lambda model, splits: QuantConfig.uniform(
        N_LAYERS, w, a, per_channel=per_channel, observer=observer, name=name
    )


def _searched(budget: float, order: str = "kl"):
    def build(model: MLP, splits: Splits) -> QuantConfig:
        return search_mixed_precision(model, splits.x_calib, splits.y_calib, budget=budget, order=order).config

    return build


RECIPES: dict[str, Callable[[MLP, Splits], QuantConfig | None]] = {
    "FP32": lambda model, splits: None,
    "W8A8 per-tensor, min-max": _uniform(8, 8, per_channel=False),
    "W8A8 per-channel, min-max": _uniform(8, 8),
    "W8A8 per-channel, MSE": _uniform(8, 8, observer="mse"),
    "W4A8 per-tensor, min-max": _uniform(4, 8, per_channel=False),
    "W4A8 per-channel, min-max": _uniform(4, 8),
    "W4A4 per-channel, min-max": _uniform(4, 4),
    "W4A4 per-channel, MSE": _uniform(4, 4, observer="mse"),
    "W3A8 per-channel, min-max": _uniform(3, 8),
    "W2A8 per-channel, min-max": _uniform(2, 8),
    "Mixed precision, searched (0.5% budget)": _searched(0.005),
    "Mixed precision, searched, KL-per-byte order (0.5% budget)": _searched(0.005, order="kl_per_byte"),
}


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def evaluate_recipe(model: MLP, splits: Splits, config: QuantConfig | None) -> dict:
    reference = model.forward(splits.x_test)
    if config is None:
        return {
            "accuracy": accuracy(reference, splits.y_test),
            "agreement": 1.0,
            "kl": 0.0,
            "bytes": 4 * model.parameters,
            "label": "FP32",
        }
    qmodel = QuantizedMLP(model, config, splits.x_calib)
    logits = qmodel.forward(splits.x_test)
    return {
        "accuracy": accuracy(logits, splits.y_test),
        "agreement": float((logits.argmax(1) == reference.argmax(1)).mean()),
        "kl": kl_divergence(reference, logits),
        "bytes": qmodel.weight_bytes(),
        "label": config.label(),
    }


def run_benchmark(seeds: int = 5, recipes: dict | None = None) -> dict:
    """Train one model per seed (own data split) and apply every recipe to it."""
    recipes = recipes or RECIPES
    per_recipe: dict[str, list[dict]] = {name: [] for name in recipes}
    integer_checks = []
    fp32_params = None
    for seed in range(seeds):
        splits = load_digits_splits(seed)
        model = train_mlp(splits.x_train, splits.y_train, seed=seed)
        fp32_params = model.parameters
        for name, build in recipes.items():
            per_recipe[name].append(evaluate_recipe(model, splits, build(model, splits)))
        integer_checks.append(integer_equivalence(model, splits))

    fp32_bytes = 4 * fp32_params
    rows = []
    for name, runs in per_recipe.items():
        rows.append(
            {
                "recipe": name,
                "config": runs[0]["label"],
                "accuracy": _stats([r["accuracy"] for r in runs]),
                "agreement": _stats([r["agreement"] for r in runs]),
                "kl": _stats([r["kl"] for r in runs]),
                "bytes": round(statistics.fmean(r["bytes"] for r in runs)),
                "compression": round(fp32_bytes / statistics.fmean(r["bytes"] for r in runs), 2),
            }
        )
    return {
        "dataset": "sklearn digits, 8x8, 10 classes; per seed: 1,091 train / 256 calibration / 450 test",
        "model": {"layers": [64, 256, 128, 10], "parameters": fp32_params, "fp32_bytes": fp32_bytes},
        "seeds": seeds,
        "recipes": rows,
        "integer_kernel": {
            "top1_agreement_with_simulation": min(c["top1_agreement"] for c in integer_checks),
            "max_code_difference_lsb": max(c["max_code_difference_lsb"] for c in integer_checks),
            "fraction_codes_differing": max(c["fraction_codes_differing"] for c in integer_checks),
            "accuracy": _stats([c["accuracy"] for c in integer_checks]),
        },
    }


def integer_equivalence(model: MLP, splits: Splits) -> dict:
    """Compare the integer-only kernel with the W8A8 simulation it is meant to reproduce."""
    config = QuantConfig.uniform(N_LAYERS, 8, 8, observer="mse")
    qmodel = QuantizedMLP(model, config, splits.x_calib)
    integer = IntegerMLP(qmodel)
    int_logits, int_codes = integer.forward(splits.x_test, return_codes=True)
    sim_logits = qmodel.forward(splits.x_test)
    sim_codes = fake_quant_codes(qmodel, splits.x_test)
    diffs = [np.abs(a - b) for a, b in zip(int_codes, sim_codes, strict=True)]
    total = sum(d.size for d in diffs)
    return {
        "top1_agreement": float((int_logits.argmax(1) == sim_logits.argmax(1)).mean()),
        "max_code_difference_lsb": int(max(d.max() for d in diffs)),
        "fraction_codes_differing": float(sum((d > 0).sum() for d in diffs) / total),
        "accuracy": accuracy(int_logits, splits.y_test),
    }


def bit_sweep(seeds: int = 5, bits=(2, 3, 4, 5, 6, 8)) -> dict:
    """Weight bit-width sweep (A8) for per-tensor vs per-channel scales, and an
    activation bit-width sweep (W8) for each calibration observer."""
    weight_rows, act_rows = [], []
    models = []
    for seed in range(seeds):
        splits = load_digits_splits(seed)
        models.append((train_mlp(splits.x_train, splits.y_train, seed=seed), splits))
    for b in bits:
        for per_channel in (False, True):
            runs = [evaluate_recipe(m, s, _uniform(b, 8, per_channel)(m, s)) for m, s in models]
            weight_rows.append(
                {
                    "bits": b,
                    "per_channel": per_channel,
                    "accuracy": statistics.fmean(r["accuracy"] for r in runs),
                    "kl": statistics.fmean(r["kl"] for r in runs),
                }
            )
        for observer in ("minmax", "percentile", "mse"):
            runs = [evaluate_recipe(m, s, _uniform(8, b, True, observer)(m, s)) for m, s in models]
            act_rows.append(
                {
                    "bits": b,
                    "observer": observer,
                    "accuracy": statistics.fmean(r["accuracy"] for r in runs),
                    "kl": statistics.fmean(r["kl"] for r in runs),
                }
            )
    fp32 = statistics.fmean(accuracy(m.forward(s.x_test), s.y_test) for m, s in models)
    return {"fp32_accuracy": fp32, "weights": weight_rows, "activations": act_rows}


def search_ablation(seeds: int = 5, budget: float = 0.005) -> dict:
    """Greedy mixed-precision search against the exhaustive optimum, per seed."""

    def summary(model: MLP, splits: Splits, config: QuantConfig) -> dict:
        result = evaluate_recipe(model, splits, config)
        return {
            "weight_bits": [s.weight_bits for s in config.layers],
            "bytes": result["bytes"],
            "test_accuracy": result["accuracy"],
            "test_kl": result["kl"],
        }

    rows = []
    for seed in range(seeds):
        splits = load_digits_splits(seed)
        model = train_mlp(splits.x_train, splits.y_train, seed=seed)
        row = {"seed": seed}
        for order in ("kl", "kl_per_byte"):
            config = search_mixed_precision(model, splits.x_calib, splits.y_calib, budget=budget, order=order).config
            row[f"greedy_{order}"] = summary(model, splits, config)
        best = exhaustive_search(model, splits.x_calib, splits.y_calib, budget=budget).config
        row["exhaustive"] = summary(model, splits, best)
        rows.append(row)
    return {
        "budget": budget,
        "quantized_forward_passes": {"greedy": 2 * N_LAYERS, "exhaustive": 2**N_LAYERS},
        "matches_exhaustive": {
            order: sum(r[f"greedy_{order}"]["weight_bits"] == r["exhaustive"]["weight_bits"] for r in rows)
            for order in ("kl", "kl_per_byte")
        },
        "seeds": rows,
    }
