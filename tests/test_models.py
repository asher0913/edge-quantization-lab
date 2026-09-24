import numpy as np
import pytest

from edge_quant.model import MLP, accuracy
from edge_quant.qmodel import IntegerMLP, QuantConfig, QuantizedMLP, fake_quant_codes
from edge_quant.search import kl_divergence, layer_sensitivity, search_mixed_precision


def test_fp32_model_trains_well(model, splits):
    assert accuracy(model.forward(splits.x_test), splits.y_test) > 0.96
    assert model.parameters == 64 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10


def test_save_load_roundtrip(model, splits, tmp_path):
    model.save(tmp_path / "m.npz")
    loaded = MLP.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(loaded.forward(splits.x_test), model.forward(splits.x_test))


def test_fp32_config_is_identity(model, splits):
    config = QuantConfig.uniform(3, None, None)
    qmodel = QuantizedMLP(model, config, splits.x_calib)
    np.testing.assert_allclose(qmodel.forward(splits.x_test), model.forward(splits.x_test), rtol=1e-6)
    assert qmodel.weight_bytes() == 4 * model.parameters


def test_w8a8_is_near_lossless(model, splits):
    qmodel = QuantizedMLP(model, QuantConfig.uniform(3, 8, 8, observer="mse"), splits.x_calib)
    logits, reference = qmodel.forward(splits.x_test), model.forward(splits.x_test)
    assert (logits.argmax(1) == reference.argmax(1)).mean() >= 0.99
    assert kl_divergence(reference, logits) < 1e-3


def test_size_accounting(model, splits):
    per_tensor = QuantizedMLP(model, QuantConfig.uniform(3, 8, 8, per_channel=False), splits.x_calib)
    weights = sum(layer.weight.size for layer in model.layers)
    biases = sum(layer.bias.size for layer in model.layers)
    assert per_tensor.weight_bytes() == weights + 4 * 3 + 4 * biases
    four_bit = QuantizedMLP(model, QuantConfig.uniform(3, 4, 8, per_channel=False), splits.x_calib)
    assert four_bit.weight_bytes() == weights // 2 + 4 * 3 + 4 * biases


def test_config_must_cover_every_layer(model, splits):
    with pytest.raises(ValueError):
        QuantizedMLP(model, QuantConfig.uniform(2, 8, 8), splits.x_calib)


@pytest.mark.parametrize("sequential", [True, False])
def test_both_calibration_modes_work(model, splits, sequential):
    qmodel = QuantizedMLP(model, QuantConfig.uniform(3, 8, 4, observer="mse"), splits.x_calib, sequential=sequential)
    assert accuracy(qmodel.forward(splits.x_test), splits.y_test) > 0.9


def test_integer_kernel_matches_simulation(model, splits):
    qmodel = QuantizedMLP(model, QuantConfig.uniform(3, 8, 8, observer="mse"), splits.x_calib)
    int_logits, int_codes = IntegerMLP(qmodel).forward(splits.x_test, return_codes=True)
    sim_codes = fake_quant_codes(qmodel, splits.x_test)
    for a, b in zip(int_codes, sim_codes, strict=True):
        assert a.dtype.kind == "i"
        assert np.abs(a - b).max() <= 1
    assert (int_logits.argmax(1) == qmodel.forward(splits.x_test).argmax(1)).all()


def test_integer_kernel_rejects_partially_quantized_models(model, splits):
    config = QuantConfig.uniform(3, 8, 8).with_layer(1, weight_bits=None)
    with pytest.raises(ValueError):
        IntegerMLP(QuantizedMLP(model, config, splits.x_calib))


def test_sensitivity_reports_every_layer(model, splits):
    rows = layer_sensitivity(model, splits.x_calib, bits=4)
    assert [r["layer"] for r in rows] == [0, 1, 2]
    assert all(r["kl"] >= 0 and 0 <= r["agreement"] <= 1 for r in rows)


@pytest.mark.parametrize("budget", [0.0, 0.005, 0.05])
def test_mixed_precision_search_respects_budget_on_calibration_data(model, splits, budget):
    result = search_mixed_precision(model, splits.x_calib, splits.y_calib, budget=budget)
    fp32 = accuracy(model.forward(splits.x_calib), splits.y_calib)
    chosen = accuracy(QuantizedMLP(model, result.config, splits.x_calib).forward(splits.x_calib), splits.y_calib)
    assert len(result.decisions) == 3
    lowered = [d for d in result.decisions if d["accepted"]]
    if lowered:
        assert fp32 - chosen <= budget + 1e-12
    assert {s.weight_bits for s in result.config.layers} <= {4, 8}


def test_generous_budget_lowers_every_layer(model, splits):
    result = search_mixed_precision(model, splits.x_calib, splits.y_calib, budget=1.0)
    assert all(s.weight_bits == 4 for s in result.config.layers)
