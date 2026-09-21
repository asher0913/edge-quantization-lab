from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from statistics import mean, median
from time import perf_counter


Matrix = list[list[float]]


def matvec(weights: Matrix, vector: list[float]) -> list[float]:
    return [sum(weight * value for weight, value in zip(row, vector)) for row in weights]


def relu(vector: list[float]) -> list[float]:
    return [max(0.0, value) for value in vector]


@dataclass(frozen=True)
class QuantizedMatrix:
    values: list[list[int]]
    scales: list[float]

    def dequantize(self) -> Matrix:
        return [[value * self.scales[index] for value in row] for index, row in enumerate(self.values)]

    @property
    def bytes(self) -> int:
        return sum(len(row) for row in self.values) + 4 * len(self.scales)


def quantize(weights: Matrix, per_channel: bool = True) -> QuantizedMatrix:
    if not weights or not weights[0]:
        raise ValueError("weights cannot be empty")
    maxima = [max(abs(value) for value in row) for row in weights]
    if not per_channel:
        maxima = [max(maxima)] * len(weights)
    scales = [maximum / 127 if maximum else 1.0 for maximum in maxima]
    values = [[max(-127, min(127, round(value / scales[index]))) for value in row] for index, row in enumerate(weights)]
    return QuantizedMatrix(values, scales)


@dataclass
class TinyMLP:
    first: Matrix
    second: Matrix

    def predict(self, vector: list[float], quantized_layers: set[int] | None = None) -> list[float]:
        quantized_layers = quantized_layers or set()
        first = quantize(self.first).dequantize() if 0 in quantized_layers else self.first
        second = quantize(self.second).dequantize() if 1 in quantized_layers else self.second
        return matvec(second, relu(matvec(first, vector)))

    def fp32_bytes(self) -> int:
        return 4 * sum(len(row) for matrix in (self.first, self.second) for row in matrix)

    def mixed_bytes(self, quantized_layers: set[int]) -> int:
        total = 0
        for index, matrix in enumerate((self.first, self.second)):
            total += quantize(matrix).bytes if index in quantized_layers else 4 * sum(len(row) for row in matrix)
        return total


def mse(left: list[float], right: list[float]) -> float:
    return mean((a - b) ** 2 for a, b in zip(left, right))


def sensitivity(model: TinyMLP, samples: list[list[float]]) -> dict[int, float]:
    result = {}
    references = [model.predict(sample) for sample in samples]
    for layer in (0, 1):
        result[layer] = mean(mse(reference, model.predict(sample, {layer})) for sample, reference in zip(samples, references))
    return result


def choose_policy(model: TinyMLP, samples: list[list[float]], max_layer_mse: float) -> set[int]:
    return {layer for layer, error in sensitivity(model, samples).items() if error <= max_layer_mse}


def fixture(seed: int = 19) -> tuple[TinyMLP, list[list[float]]]:
    rng = random.Random(seed)
    matrix = lambda rows, cols: [[rng.uniform(-1.5, 1.5) for _ in range(cols)] for _ in range(rows)]
    return TinyMLP(matrix(12, 8), matrix(4, 12)), [[rng.uniform(-1, 1) for _ in range(8)] for _ in range(48)]


def benchmark(model: TinyMLP, samples: list[list[float]], policy: set[int]) -> dict[str, object]:
    references = [model.predict(sample) for sample in samples]
    outputs = [model.predict(sample, policy) for sample in samples]
    agreement = mean(max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__) for a, b in zip(references, outputs))
    timings = []
    for _ in range(7):
        start = perf_counter()
        for sample in samples:
            model.predict(sample, policy)
        timings.append((perf_counter() - start) * 1000)
    return {
        "quantized_layers": sorted(policy),
        "output_mse": mean(mse(a, b) for a, b in zip(references, outputs)),
        "top1_agreement": agreement,
        "fp32_bytes": model.fp32_bytes(),
        "mixed_bytes": model.mixed_bytes(policy),
        "median_batch_ms": median(timings),
    }


def demo() -> dict[str, object]:
    model, samples = fixture()
    errors = sensitivity(model, samples[:24])
    policy = choose_policy(model, samples[:24], max(errors.values()))
    report = benchmark(model, samples[24:], policy)
    report["layer_sensitivity"] = errors
    return report


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))
