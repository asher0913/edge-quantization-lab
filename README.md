# Edge Quantization Lab

[![CI](https://github.com/asher0913/edge-quantization-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/asher0913/edge-quantization-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Post-training quantization implemented from first principles in NumPy: symmetric and affine
quantizers, per-tensor and per-channel weight scales, three activation-calibration observers,
per-layer sensitivity analysis, mixed-precision search under an accuracy budget, and an
**integer-only inference kernel** (int8 × int8 → int32 with fixed-point requantization) that is
checked against the floating-point simulation.

Everything is measured on a real trained network, not random matrices: a 64-256-128-10 MLP
(50,826 parameters) on the 8×8 handwritten-digits set bundled with scikit-learn, over **5 seeds**,
each with its own train / calibration / test split. Every quantization decision uses only the
256-image calibration set; the test set is used for reporting.

## Results

| Recipe | Test accuracy % | Top-1 agreement with FP32 | Output KL | Size | Compression |
|---|---:|---:|---:|---:|---:|
| FP32 | 98.18 ± 0.37 | 100% | 0 | 198.5 KiB | 1.0× |
| W8A8, per-tensor, min-max | 98.13 ± 0.34 | 99.96% | 3×10⁻⁵ | 50.8 KiB | 3.9× |
| W8A8, per-channel, MSE | 98.13 ± 0.37 | 99.96% | 2×10⁻⁵ | 52.3 KiB | 3.8× |
| W4A8, per-tensor | 97.87 ± 0.40 | 99.20% | 6.3×10⁻³ | 26.2 KiB | 7.6× |
| W4A8, per-channel | 97.69 ± 0.34 | 99.24% | 4.2×10⁻³ | 27.7 KiB | 7.2× |
| W4A4, per-channel | 97.64 ± 0.25 | 99.16% | 5.7×10⁻³ | 27.7 KiB | 7.2× |
| **Mixed precision, searched (0.5% budget)** | **97.82 ± 0.40** | **99.47%** | **3.6×10⁻³** | **31.0 KiB** | **6.4×** |
| W3A8, per-channel | 97.07 ± 0.91 | 98.31% | 2.3×10⁻² | 21.5 KiB | 9.2× |
| W2A8, per-channel | 66.89 ± 13.23 | 67.47% | 1.06 | 15.4 KiB | 12.9× |

*Mean ± std over 5 seeds (450 test images each). "Output KL" is KL(FP32 ‖ quantized) between softmax
outputs; it separates recipes long before accuracy does. Size includes int32 biases and one FP32
scale per tensor or channel. Reproduce with `edge-quant benchmark --out results/benchmark.json`
(about 3 s on a laptop).*

**Integer-only kernel.** Running W8A8 with nothing but integer arithmetic after the input layer
reproduces the simulated model's top-1 prediction on 100% of test images across all 5 seeds. The
hidden-layer codes differ in 0.0025% of positions, never by more than 1 LSB; those differences come
from the fixed-point rescale rounding exact .5 ties differently from floating point.

![Weight and activation bit-width sweeps](docs/bit_sweep.png)

What the sweeps show:

- **8 bits is free, 4 bits is cheap, 2 bits is a cliff.** Accuracy is flat down to 5-bit weights,
  loses 0.3–0.5 points at 4 bits and 1.1 at 3 bits, then collapses at 2 bits.
- **Per-channel scales cut output KL by a third to a half at every bit-width** and are the difference
  between a usable and an unusable model at 2 bits (66.9% vs 32.0% accuracy). At 4 bits the
  accuracy gap is inside seed-to-seed noise; KL still tells them apart.
- **Calibration matters when activation levels are scarce.** An MSE-optimal clipping range lowers
  output KL by 63% versus min-max at 2-bit activations, 32% at 3 bits and 30% at 4 bits; at 8 bits
  all three observers are equivalent.
- **Mixed precision buys most of INT4's size at a fraction of its error.** The search keeps the
  input layer at 8 bits on 2 of 5 seeds and the output layer at 8 bits on 1, and otherwise drops
  to 4 bits, landing at 6.4× compression with lower KL than any uniform 4-bit recipe.

## How it works

```mermaid
flowchart LR
    M[Trained FP32 MLP] --> C[Calibration batch<br/>256 images]
    C --> O["Observers<br/>min-max · percentile · MSE"]
    M --> W["Weight quantizer<br/>symmetric, per-tensor / per-channel"]
    O --> Q[Fake-quantized model]
    W --> Q
    Q --> S["Sensitivity<br/>KL per layer at INT4"]
    S --> G["Greedy bit allocation<br/>≤ budget on calibration set"]
    Q --> I["Integer-only kernel<br/>int32 acc, fixed-point rescale"]
    G --> R[Report on held-out test set]
    I --> R
```

- **Quantizers** (`quant.py`). Weights: symmetric signed, restricted range [−127, 127] so that zero
  and ±max are exact. Activations: asymmetric unsigned with a zero point, which suits post-ReLU
  tensors. Biases live on the accumulator grid (`scale_in × scale_w`) as saturating int32.
- **Observers.** Min-max uses the observed extremes; percentile clips at the 0.01st and 99.99th
  percentiles; MSE searches 121 log-spaced clipping ratios for the one that minimises
  reconstruction error on the calibration tensor.
- **Sequential calibration** (`qmodel.py`). Each layer's activation range is calibrated on the
  output of the already-quantized layers before it, so ranges match what the deployed model sees.
- **Integer kernel.** Each layer computes `acc = (q_in − z_in) · W_q + b_q` in int64 (int32 in
  hardware), then applies the per-channel multiplier `M = s_in·s_w / s_out` as a 31-bit mantissa
  and a rounding right shift, the representation used by gemmlowp and TFLite. The ReLU is the lower
  clamp at the output zero point.
- **Mixed-precision search** (`search.py`). Rank layers by INT4 sensitivity (softmax KL on
  calibration data, larger layers first on ties), then lower each one to INT4 only if calibration
  accuracy stays within the budget of FP32.

## A deployment bug this caught

The first version of the integer kernel hit invalid integer casts on some seeds. The cause was
**dead ReLU units**: weight decay drove every weight of a few channels to about 10⁻²⁴, so the per-channel scale became
10⁻²⁶ and the bias, stored as `bias / (s_in · s_w)`, overflowed int64. The fix mirrors production
runtimes: a floor on each channel's weight range and int32 saturation of quantized biases. Both
are covered by tests.

## Usage

```bash
pip install -e '.[dev]'

edge-quant benchmark --seeds 5 --out results/benchmark.json
edge-quant sweep --seeds 5 --out results/bit_sweep.json
edge-quant sensitivity --bits 4
edge-quant search --budget 0.005
python scripts/make_figures.py     # needs matplotlib
```

```python
from edge_quant.model import load_digits_splits, train_mlp
from edge_quant.qmodel import QuantConfig, QuantizedMLP, IntegerMLP

splits = load_digits_splits(seed=0)
model = train_mlp(splits.x_train, splits.y_train, seed=0)
qmodel = QuantizedMLP(model, QuantConfig.uniform(3, 8, 8, observer="mse"), splits.x_calib)
logits = IntegerMLP(qmodel).forward(splits.x_test)  # integer arithmetic end to end
```

## Tests

`pytest -q` runs 40 tests in about 3 seconds: round-trip error bounded by half a quantization step,
per-channel never worse than per-tensor, zero exactly representable, observer behaviour on an
outlier, fixed-point multiplier accuracy to 2⁻³⁰, integer rounding against float rounding,
byte accounting, the integer kernel within 1 LSB of simulation, rejection of partially quantized
models in the integer path, and the mixed-precision search staying within its budget.

## Limitations

- The model is deliberately small so the whole study runs on a CPU in seconds. Convolutions,
  batch-norm folding, depthwise layers and attention bring quantization issues (outlier channels,
  large activation ranges) that an MLP on 8×8 digits does not exhibit.
- Latency is not reported. NumPy has no int8 GEMM, so timing the integer kernel here would say
  nothing about an NPU, DSP or TensorRT engine; the value of this code is numerical fidelity.
- Only post-training quantization is covered. Quantization-aware training is the usual next step
  for the 3- and 2-bit regimes and is not implemented here.
- With 450 test images, one image is 0.22 points, so accuracy differences under about 0.4 points
  are within seed noise; the KL and agreement columns are the more sensitive signals.

## License

MIT
