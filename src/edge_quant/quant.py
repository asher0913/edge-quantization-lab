"""Uniform affine quantization, weight scales and activation observers.

A real value ``r`` is represented by an integer ``q`` with

    r ≈ scale * (q - zero_point),    q ∈ [qmin, qmax]

Weights use symmetric signed quantization (``zero_point = 0``), per tensor or
per output channel. Activations use asymmetric unsigned quantization, whose
range comes from a calibration *observer*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QParams:
    scale: np.ndarray  # scalar, or one entry per output channel
    zero_point: np.ndarray  # same shape as scale, integer valued
    bits: int
    signed: bool

    @property
    def qmin(self) -> int:
        return -(2 ** (self.bits - 1)) + 1 if self.signed else 0  # symmetric range for signed

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1 if self.signed else 2**self.bits - 1

    def _broadcast(self, x: np.ndarray, value: np.ndarray) -> np.ndarray:
        # Per-channel parameters index the first axis (output channel) of a weight.
        return value.reshape(-1, *([1] * (x.ndim - 1))) if value.ndim == 1 and value.size > 1 else value

    def quantize(self, x: np.ndarray) -> np.ndarray:
        scale, zp = self._broadcast(x, self.scale), self._broadcast(x, self.zero_point)
        return np.clip(np.round(x / scale) + zp, self.qmin, self.qmax).astype(np.int64)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        scale, zp = self._broadcast(q, self.scale), self._broadcast(q, self.zero_point)
        return ((q - zp) * scale).astype(np.float32)

    def fake_quantize(self, x: np.ndarray) -> np.ndarray:
        return self.dequantize(self.quantize(x))


MIN_WEIGHT_RANGE = 1e-6
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def weight_qparams(weight: np.ndarray, bits: int, per_channel: bool) -> QParams:
    """Symmetric scales: the largest magnitude maps to ``qmax``.

    Channels whose weights have collapsed to ~0 (dead ReLU units after weight
    decay) get a floor on their range. Without it the scale can reach 1e-30
    and the bias, stored on the accumulator scale, overflows int32.
    """
    if bits < 2:
        raise ValueError("need at least 2 bits")
    qmax = 2 ** (bits - 1) - 1
    max_abs = np.abs(weight).max(axis=1) if per_channel else np.array(np.abs(weight).max())
    scale = (np.maximum(max_abs, MIN_WEIGHT_RANGE) / qmax).astype(np.float64)
    return QParams(scale, np.zeros_like(scale, dtype=np.int64), bits, signed=True)


def quantize_bias(bias: np.ndarray, accumulator_scale: np.ndarray) -> np.ndarray:
    """int32 bias on the ``scale_in * scale_w`` grid, saturated like real accelerators."""
    return np.clip(np.round(bias / accumulator_scale), INT32_MIN, INT32_MAX).astype(np.int64)


def activation_qparams(low: float, high: float, bits: int) -> QParams:
    """Asymmetric unsigned parameters covering ``[low, high]`` (always including 0)."""
    low, high = min(low, 0.0), max(high, 0.0)
    qmax = 2**bits - 1
    scale = (high - low) / qmax if high > low else 1.0
    zero_point = int(np.clip(round(-low / scale), 0, qmax))
    return QParams(np.array(scale), np.array(zero_point), bits, signed=False)


def observe_range(values: np.ndarray, method: str, bits: int) -> tuple[float, float]:
    """Choose the clipping range for an activation tensor from calibration data.

    ``minmax``      the observed extremes; one outlier can waste most of the grid.
    ``percentile``  the 0.01st / 99.99th percentiles; clips rare outliers.
    ``mse``         the clipping range that minimises quantization MSE on the
                    calibration values, found by a log-spaced 1-D search over
                    the clipping ratio (activations here are post-ReLU or pixels).
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    low, high = float(values.min()), float(values.max())
    if method == "minmax":
        return low, high
    if method == "percentile":
        return float(np.percentile(values, 0.01)), float(np.percentile(values, 99.99))
    if method == "mse":
        if high <= low:
            return low, high
        best, best_err = (low, high), np.inf
        for ratio in np.geomspace(0.01, 1.0, 121):
            candidate = (low * ratio, high * ratio) if low < 0 else (low, high * ratio)
            qp = activation_qparams(*candidate, bits)
            err = float(np.mean((qp.fake_quantize(values) - values) ** 2))
            if err < best_err:
                best, best_err = candidate, err
        return best
    raise ValueError(f"unknown observer {method!r}")


def quantize_multiplier(real_multiplier: float) -> tuple[int, int]:
    """Represent a positive real ``M`` as ``m0 * 2**-shift`` with a 31-bit ``m0``.

    This is how integer-only runtimes (gemmlowp, TFLite) apply the rescale
    ``scale_in * scale_w / scale_out`` without floating point.
    """
    if not real_multiplier > 0:
        raise ValueError("multiplier must be positive")
    mantissa, exponent = np.frexp(real_multiplier)  # real = mantissa * 2**exponent, mantissa in [0.5, 1)
    m0 = int(round(mantissa * (1 << 31)))
    shift = 31 - int(exponent)
    if m0 == 1 << 31:  # rounding overflow
        m0 //= 2
        shift -= 1
    if shift < 1:
        raise ValueError("multiplier too large for a right-shift representation")
    return m0, shift


def rounding_right_shift(values: np.ndarray, m0: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """``round(values * m0 / 2**shift)`` in 64-bit integer arithmetic, ties away from zero."""
    product = values.astype(np.int64) * m0.astype(np.int64)
    half = np.left_shift(np.int64(1), shift - 1)
    return np.where(product >= 0, (product + half) >> shift, -((-product + half) >> shift))
