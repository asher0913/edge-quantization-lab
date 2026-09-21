import unittest

from edge_quantization_lab.core import benchmark, choose_policy, fixture, quantize, sensitivity


class QuantizationTests(unittest.TestCase):
    def test_quantization_bounds(self):
        result = quantize([[0.0, 3.0, -4.0]])
        self.assertTrue(all(-127 <= value <= 127 for row in result.values for value in row))

    def test_per_channel_reconstructs_shape(self):
        weights = [[1.0, 2.0], [-3.0, 4.0]]
        self.assertEqual([len(row) for row in quantize(weights).dequantize()], [2, 2])

    def test_policy_respects_threshold(self):
        model, samples = fixture()
        errors = sensitivity(model, samples[:10])
        policy = choose_policy(model, samples[:10], min(errors.values()))
        self.assertGreaterEqual(len(policy), 1)

    def test_benchmark_reports_size(self):
        model, samples = fixture()
        report = benchmark(model, samples[:10], {0, 1})
        self.assertLess(report["mixed_bytes"], report["fp32_bytes"])
        self.assertGreaterEqual(report["top1_agreement"], 0)


if __name__ == "__main__":
    unittest.main()
