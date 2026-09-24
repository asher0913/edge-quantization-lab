"""Redraw docs/bit_sweep.png from results/bit_sweep.json (run `edge-quant sweep --out ...` first)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def bit_sweep() -> None:
    sweep = json.loads((ROOT / "results" / "bit_sweep.json").read_text())
    fig, (acc_ax, w_ax, a_ax) = plt.subplots(1, 3, figsize=(13, 3.6))
    for per_channel, label in ((False, "per-tensor"), (True, "per-channel")):
        rows = [r for r in sweep["weights"] if r["per_channel"] == per_channel]
        acc_ax.plot([r["bits"] for r in rows], [100 * r["accuracy"] for r in rows], marker="o", label=label)
        w_ax.plot([r["bits"] for r in rows], [r["kl"] for r in rows], marker="o", label=label)
    acc_ax.axhline(100 * sweep["fp32_accuracy"], color="grey", ls="--", lw=1, label="FP32")
    acc_ax.set(title="Weight bits (A8): accuracy", xlabel="weight bits", ylabel="test accuracy %", ylim=(60, 100))
    w_ax.set(title="Weight bits (A8): output KL", xlabel="weight bits", ylabel="KL(FP32 ‖ quantized)", yscale="log")
    for observer, label in (("minmax", "min-max"), ("percentile", "percentile"), ("mse", "MSE-optimal")):
        rows = [r for r in sweep["activations"] if r["observer"] == observer]
        a_ax.plot([r["bits"] for r in rows], [r["kl"] for r in rows], marker="o", label=label)
    a_ax.set(
        title="Activation bits (W8): calibration", xlabel="activation bits", ylabel="KL(FP32 ‖ quantized)", yscale="log"
    )
    for ax in (acc_ax, w_ax, a_ax):
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "bit_sweep.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    bit_sweep()
