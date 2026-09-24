import json

from edge_quant.benchmark import run_benchmark
from edge_quant.cli import main


def test_benchmark_structure_and_ordering():
    report = run_benchmark(seeds=1)
    rows = {r["recipe"]: r for r in report["recipes"]}
    assert rows["FP32"]["compression"] == 1.0
    assert rows["W8A8 per-tensor, min-max"]["compression"] > 3.5
    assert rows["W4A8 per-tensor, min-max"]["compression"] > 7
    assert rows["W8A8 per-channel, MSE"]["kl"]["mean"] < rows["W4A8 per-channel, min-max"]["kl"]["mean"]
    assert rows["W2A8 per-channel, min-max"]["accuracy"]["mean"] < rows["W8A8 per-channel, MSE"]["accuracy"]["mean"]
    assert report["integer_kernel"]["top1_agreement_with_simulation"] == 1.0
    assert report["integer_kernel"]["max_code_difference_lsb"] <= 1


def test_cli_commands(tmp_path, capsys):
    out = tmp_path / "bench.json"
    assert main(["benchmark", "--seeds", "1", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["seeds"] == 1
    assert main(["sensitivity", "--bits", "4"]) == 0
    assert main(["search", "--budget", "0.01"]) == 0
    assert "Mixed" in capsys.readouterr().out
