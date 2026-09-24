"""Post-training quantized models: simulated (fake-quant) and integer-only."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .model import MLP
from .quant import (
    QParams,
    activation_qparams,
    observe_range,
    quantize_bias,
    quantize_multiplier,
    rounding_right_shift,
    weight_qparams,
)


@dataclass(frozen=True)
class LayerSpec:
    """Precision of one dense layer. ``None`` keeps that tensor in FP32."""

    weight_bits: int | None = 8
    act_bits: int | None = 8  # precision of this layer's *input* activation
    per_channel: bool = True


@dataclass(frozen=True)
class QuantConfig:
    layers: tuple[LayerSpec, ...]
    observer: str = "minmax"
    name: str = ""

    @classmethod
    def uniform(
        cls,
        n_layers: int,
        weight_bits: int | None,
        act_bits: int | None,
        *,
        per_channel: bool = True,
        observer: str = "minmax",
        name: str = "",
    ) -> QuantConfig:
        spec = LayerSpec(weight_bits, act_bits, per_channel)
        return cls(tuple(spec for _ in range(n_layers)), observer, name)

    def with_layer(self, index: int, **changes) -> QuantConfig:
        layers = list(self.layers)
        layers[index] = replace(layers[index], **changes)
        return replace(self, layers=tuple(layers))

    def label(self) -> str:
        if self.name:
            return self.name
        return " / ".join(f"W{s.weight_bits or 32}A{s.act_bits or 32}" for s in self.layers)


@dataclass
class _QLayer:
    weight: np.ndarray  # FP32 weight (kept for FP32 layers)
    bias: np.ndarray
    weight_q: np.ndarray | None
    weight_qp: QParams | None
    act_qp: QParams | None


@dataclass
class QuantizedMLP:
    """Fake-quantized MLP: every tensor is rounded to its integer grid and back.

    Activation ranges are calibrated on a small calibration batch. With
    ``sequential=True`` (the default) each layer is calibrated on the
    activations produced by the *already quantized* earlier layers, so the
    ranges match what the deployed model will actually see.
    """

    model: MLP
    config: QuantConfig
    calibration: np.ndarray
    sequential: bool = True
    layers: list[_QLayer] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.config.layers) != len(self.model.layers):
            raise ValueError("config must describe every layer")
        self.layers = []
        x = self.calibration
        fp32_trace = self.model.forward_trace(self.calibration)
        for i, (dense, spec) in enumerate(zip(self.model.layers, self.config.layers, strict=True)):
            source = x if self.sequential else fp32_trace[i]
            act_qp = None
            if spec.act_bits is not None:
                act_qp = activation_qparams(*observe_range(source, self.config.observer, spec.act_bits), spec.act_bits)
            weight_q = weight_qp = None
            if spec.weight_bits is not None:
                weight_qp = weight_qparams(dense.weight, spec.weight_bits, spec.per_channel)
                weight_q = weight_qp.quantize(dense.weight)
            layer = _QLayer(dense.weight, dense.bias, weight_q, weight_qp, act_qp)
            self.layers.append(layer)
            x = self._layer_forward(layer, x, last=i == len(self.model.layers) - 1)

    def _effective_bias(self, layer: _QLayer) -> np.ndarray:
        if layer.act_qp is None or layer.weight_qp is None:
            return layer.bias
        # Integer runtimes store the bias as int32 on the accumulator's scale.
        scale = layer.act_qp.scale * layer.weight_qp.scale
        return (quantize_bias(layer.bias, scale) * scale).astype(np.float32)

    def _layer_forward(self, layer: _QLayer, x: np.ndarray, last: bool) -> np.ndarray:
        if layer.act_qp is not None:
            x = layer.act_qp.fake_quantize(x)
        weight = layer.weight_qp.dequantize(layer.weight_q) if layer.weight_qp is not None else layer.weight
        out = x @ weight.T + self._effective_bias(layer)
        return out if last else np.maximum(out, 0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, layer in enumerate(self.layers):
            x = self._layer_forward(layer, x, last=i == len(self.layers) - 1)
        return x

    def weight_bytes(self) -> int:
        """Storage for weights, scales and biases (int32 bias when both sides are quantized)."""
        total = 0
        for layer in self.layers:
            if layer.weight_qp is None:
                total += 4 * (layer.weight.size + layer.bias.size)
                continue
            total += math.ceil(layer.weight.size * layer.weight_qp.bits / 8)
            total += 4 * layer.weight_qp.scale.size + 4 * layer.bias.size
        return total


class IntegerMLP:
    """Integer-only inference for a fully quantized model.

    Only the network input is quantized in floating point. Every layer then
    computes ``int32 acc = (q_in - z_in) · W_q + b_q`` and requantizes with a
    per-channel fixed-point multiplier and a rounding right shift, exactly as
    an int8 accelerator would. The final layer's accumulator is dequantized to
    produce logits.
    """

    def __init__(self, qmodel: QuantizedMLP) -> None:
        if any(layer.weight_qp is None or layer.act_qp is None for layer in qmodel.layers):
            raise ValueError("integer inference needs every weight and activation quantized")
        self.qmodel = qmodel
        self.plan = []
        n = len(qmodel.layers)
        for i, layer in enumerate(qmodel.layers):
            s_in, z_in = float(layer.act_qp.scale), int(layer.act_qp.zero_point)
            s_w = np.broadcast_to(layer.weight_qp.scale, (layer.weight.shape[0],)).astype(np.float64)
            acc_scale = s_in * s_w
            bias_q = quantize_bias(layer.bias, acc_scale)
            step = {"w": layer.weight_q.astype(np.int64), "b": bias_q, "z_in": z_in, "acc_scale": acc_scale}
            if i < n - 1:
                nxt = qmodel.layers[i + 1].act_qp
                pairs = [quantize_multiplier(m) for m in acc_scale / float(nxt.scale)]
                step["m0"] = np.array([p[0] for p in pairs], dtype=np.int64)
                step["shift"] = np.array([p[1] for p in pairs], dtype=np.int64)
                step["z_out"], step["qmax"] = int(nxt.zero_point), nxt.qmax
            self.plan.append(step)

    def forward(self, x: np.ndarray, *, return_codes: bool = False):
        q = self.qmodel.layers[0].act_qp.quantize(x)
        codes = [q]
        for step in self.plan[:-1]:
            acc = (q - step["z_in"]) @ step["w"].T + step["b"]
            scaled = rounding_right_shift(acc, step["m0"], step["shift"]) + step["z_out"]
            q = np.clip(scaled, step["z_out"], step["qmax"])  # lower clamp at z_out is the ReLU
            codes.append(q)
        last = self.plan[-1]
        acc = (q - last["z_in"]) @ last["w"].T + last["b"]
        logits = (acc * last["acc_scale"]).astype(np.float32)
        return (logits, codes) if return_codes else logits


def fake_quant_codes(qmodel: QuantizedMLP, x: np.ndarray) -> list[np.ndarray]:
    """Integer codes of each layer input under the simulated model, for comparison."""
    codes = []
    for i, layer in enumerate(qmodel.layers):
        codes.append(layer.act_qp.quantize(x))
        x = qmodel._layer_forward(layer, x, last=i == len(qmodel.layers) - 1)
    return codes
