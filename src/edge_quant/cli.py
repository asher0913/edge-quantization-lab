"""``edge-quant benchmark | sweep | sensitivity | search | ablate-search``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import bit_sweep, run_benchmark, search_ablation
from .model import load_digits_splits, train_mlp
from .search import layer_sensitivity, search_mixed_precision


def _fmt(stat: dict, scale: float = 100.0, digits: int = 2) -> str:
    return f"{scale * stat['mean']:.{digits}f} ± {scale * stat['std']:.{digits}f}"


def _benchmark(args) -> int:
    report = run_benchmark(seeds=args.seeds)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['model']['parameters']:,} parameters, FP32 {report['model']['fp32_bytes']:,} bytes\n")
    print("| Recipe | Test accuracy % | Top-1 agreement % | KL(FP32‖Q) | Bytes | Compression |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in report["recipes"]:
        print(
            f"| {row['recipe']} | {_fmt(row['accuracy'])} | {100 * row['agreement']['mean']:.2f} | "
            f"{row['kl']['mean']:.5f} | {row['bytes']:,} | {row['compression']}× |"
        )
    print("\nInteger-only kernel vs simulation:", json.dumps(report["integer_kernel"], indent=2))
    return 0


def _sweep(args) -> int:
    report = bit_sweep(seeds=args.seeds)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def _sensitivity(args) -> int:
    splits = load_digits_splits(args.seed)
    model = train_mlp(splits.x_train, splits.y_train, seed=args.seed)
    for row in layer_sensitivity(model, splits.x_calib, args.bits):
        print(f"layer {row['layer']} {row['shape']}: KL={row['kl']:.5f} agreement={row['agreement']:.4f}")
    return 0


def _search(args) -> int:
    splits = load_digits_splits(args.seed)
    model = train_mlp(splits.x_train, splits.y_train, seed=args.seed)
    result = search_mixed_precision(model, splits.x_calib, splits.y_calib, budget=args.budget)
    print(result.config.label())
    for decision in result.decisions:
        print(json.dumps(decision))
    return 0


def _ablate_search(args) -> int:
    report = search_ablation(seeds=args.seeds, budget=args.budget)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("| Seed | Greedy, KL order | Greedy, KL per byte | Exhaustive |")
    print("|---:|---|---|---|")
    for row in report["seeds"]:
        cells = [row[k] for k in ("greedy_kl", "greedy_kl_per_byte", "exhaustive")]
        print(f"| {row['seed']} | " + " | ".join(f"{c['weight_bits']} {c['bytes']:,} B" for c in cells) + " |")
    n = len(report["seeds"])
    for order, hits in report["matches_exhaustive"].items():
        print(f"greedy ({order}) matches exhaustive on {hits} of {n} seeds")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edge-quant", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark", help="every recipe, averaged over seeds")
    bench.add_argument("--seeds", type=int, default=5)
    bench.add_argument("--out", help="write the JSON report here")
    bench.set_defaults(func=_benchmark)
    sweep = sub.add_parser("sweep", help="weight and activation bit-width sweeps")
    sweep.add_argument("--seeds", type=int, default=5)
    sweep.add_argument("--out")
    sweep.set_defaults(func=_sweep)
    sens = sub.add_parser("sensitivity", help="per-layer damage at a given weight bit-width")
    sens.add_argument("--bits", type=int, default=4)
    sens.add_argument("--seed", type=int, default=0)
    sens.set_defaults(func=_sensitivity)
    search = sub.add_parser("search", help="mixed-precision search under an accuracy budget")
    search.add_argument("--budget", type=float, default=0.005)
    search.add_argument("--seed", type=int, default=0)
    search.set_defaults(func=_search)
    ablate = sub.add_parser("ablate-search", help="greedy mixed-precision search vs the exhaustive optimum")
    ablate.add_argument("--seeds", type=int, default=5)
    ablate.add_argument("--budget", type=float, default=0.005)
    ablate.add_argument("--out")
    ablate.set_defaults(func=_ablate_search)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
