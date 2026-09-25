import pytest

from edge_quant.benchmark import search_ablation
from edge_quant.model import accuracy
from edge_quant.qmodel import QuantizedMLP
from edge_quant.search import exhaustive_search, search_mixed_precision

BUDGET = 0.005


def _calib_accuracy(model, splits, config):
    return accuracy(QuantizedMLP(model, config, splits.x_calib).forward(splits.x_calib), splits.y_calib)


def _bytes(model, splits, config):
    return QuantizedMLP(model, config, splits.x_calib).weight_bytes()


def test_exhaustive_search_tries_every_assignment_and_respects_budget(model, splits):
    result = exhaustive_search(model, splits.x_calib, splits.y_calib, budget=BUDGET)
    assert len(result.decisions) == 2 ** len(model.layers)
    fp32 = accuracy(model.forward(splits.x_calib), splits.y_calib)
    assert fp32 - _calib_accuracy(model, splits, result.config) <= BUDGET
    feasible = [d["bytes"] for d in result.decisions if d["within_budget"]]
    assert _bytes(model, splits, result.config) == min(feasible)


@pytest.mark.parametrize("order", ["kl", "kl_per_byte"])
def test_greedy_is_never_smaller_than_exhaustive(model, splits, order):
    greedy = search_mixed_precision(model, splits.x_calib, splits.y_calib, budget=BUDGET, order=order).config
    best = exhaustive_search(model, splits.x_calib, splits.y_calib, budget=BUDGET).config
    assert _bytes(model, splits, greedy) >= _bytes(model, splits, best)


def test_unknown_order_is_rejected(model, splits):
    with pytest.raises(ValueError):
        search_mixed_precision(model, splits.x_calib, splits.y_calib, order="random")


def test_search_ablation_reports_both_orders():
    report = search_ablation(seeds=1)
    assert set(report["matches_exhaustive"]) == {"kl", "kl_per_byte"}
    row = report["seeds"][0]
    for key in ("greedy_kl", "greedy_kl_per_byte", "exhaustive"):
        assert row[key]["bytes"] >= row["exhaustive"]["bytes"]
