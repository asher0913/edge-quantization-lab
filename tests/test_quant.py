import numpy as np
import pytest

from edge_quant.quant import (
    INT32_MAX,
    activation_qparams,
    observe_range,
    quantize_bias,
    quantize_multiplier,
    rounding_right_shift,
    weight_qparams,
)

rng = np.random.default_rng(0)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("per_channel", [False, True])
def test_weight_roundtrip_error_is_at_most_half_a_step(bits, per_channel):
    weight = rng.standard_normal((16, 32)).astype(np.float32)
    qp = weight_qparams(weight, bits, per_channel)
    q = qp.quantize(weight)
    assert q.min() >= -(2 ** (bits - 1)) + 1 and q.max() <= 2 ** (bits - 1) - 1
    step = qp.scale.reshape(-1, 1) if per_channel else qp.scale
    assert np.all(np.abs(qp.dequantize(q) - weight) <= step / 2 + 1e-7)
    assert np.all(qp.zero_point == 0)
    assert qp.scale.shape == ((16,) if per_channel else ())


def test_per_channel_never_loses_to_per_tensor():
    weight = rng.standard_normal((8, 64)) * np.linspace(0.01, 3, 8)[:, None]
    tensor, channel = weight_qparams(weight, 4, False), weight_qparams(weight, 4, True)
    err_t = np.mean((tensor.fake_quantize(weight) - weight) ** 2)
    err_c = np.mean((channel.fake_quantize(weight) - weight) ** 2)
    assert err_c < err_t


def test_dead_channel_gets_a_scale_floor_and_bias_saturates():
    weight = np.zeros((2, 4))
    weight[1] = 1.0
    qp = weight_qparams(weight, 8, per_channel=True)
    assert qp.scale[0] > 0 and np.isfinite(qp.scale[0])
    assert quantize_bias(np.array([1e30]), np.array([1e-30]))[0] == INT32_MAX


def test_activation_zero_is_exactly_representable():
    qp = activation_qparams(-0.37, 2.9, 8)
    assert qp.fake_quantize(np.array([0.0]))[0] == 0.0
    assert 0 <= int(qp.zero_point) <= 255


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_observers_on_an_outlier(bits):
    values = np.abs(rng.standard_normal(5000))
    values[0] = 50.0  # one outlier

    def mse(high):
        qp = activation_qparams(0.0, high, bits)
        return np.mean((qp.fake_quantize(values) - values) ** 2)

    _, hi_mm = observe_range(values, "minmax", bits)
    _, hi_pct = observe_range(values, "percentile", bits)
    _, hi_mse = observe_range(values, "mse", bits)
    assert hi_mm == pytest.approx(50.0)
    assert hi_pct < hi_mm
    assert mse(hi_mse) <= mse(hi_mm)  # min-max is one of the candidates the search considers


def test_mse_observer_clips_only_when_levels_are_scarce():
    values = np.abs(rng.standard_normal(5000))
    values[0] = 50.0
    # With 4 levels, sacrificing the outlier buys resolution for everything else;
    # with 256 levels the grid is fine enough that keeping the outlier is cheaper.
    assert observe_range(values, "mse", 2)[1] < 10
    assert observe_range(values, "mse", 8)[1] == pytest.approx(50.0)


def test_unknown_observer():
    with pytest.raises(ValueError):
        observe_range(np.ones(3), "magic", 8)


@pytest.mark.parametrize("real", [1e-9, 3.7e-4, 0.123, 0.5, 0.9999999, 1.5, 1234.5])
def test_fixed_point_multiplier_is_accurate(real):
    m0, shift = quantize_multiplier(real)
    assert 2**30 <= m0 < 2**31
    assert abs(m0 / 2**shift - real) / real < 2**-30


def test_multiplier_validation():
    with pytest.raises(ValueError):
        quantize_multiplier(0.0)
    with pytest.raises(ValueError):
        quantize_multiplier(2.0**40)


def test_rounding_right_shift_matches_float_rounding():
    values = rng.integers(-(2**24), 2**24, size=20000)
    m0, shift = quantize_multiplier(0.0137)
    got = rounding_right_shift(values, np.array(m0), np.array(shift))
    expected = np.round(values * (m0 / 2**shift))
    # identical except for exact .5 ties, which integer code rounds away from zero
    assert np.max(np.abs(got - expected)) <= 1
    assert np.mean(got != expected) < 1e-3
