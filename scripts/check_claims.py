"""Check the README's claims against a fresh run, with explicit tolerances.

    python scripts/check_claims.py runs/demo

Training uses BLAS matrix products, whose summation order differs between libraries (Accelerate
on macOS, OpenBLAS on Linux). The trained weights, and so every downstream number, can differ in
the last bits across platforms. The committed results/ were produced on macOS arm64. On other
platforms this script checks what the README actually claims, rather than bit equality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []


def check(ok: bool, message: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + message)
    if not ok:
        FAILURES.append(message)


def main(fresh_dir: str) -> int:
    fresh = {n: json.loads((Path(fresh_dir) / f"{n}.json").read_text()) for n in ("benchmark", "search_ablation")}
    committed = {n: json.loads((ROOT / "results" / f"{n}.json").read_text()) for n in ("benchmark", "search_ablation")}
    rows = {r["recipe"]: r for r in fresh["benchmark"]["recipes"]}
    old = {r["recipe"]: r for r in committed["benchmark"]["recipes"]}

    print("Recipe accuracy within 0.5 points of the committed mean (2 points at W2, where seed std is 13):")
    for name, row in rows.items():
        tol = 0.02 if name.startswith("W2") else 0.005
        delta = row["accuracy"]["mean"] - old[name]["accuracy"]["mean"]
        check(abs(delta) <= tol, f"{name}: {100 * row['accuracy']['mean']:.2f}% ({100 * delta:+.2f})")

    print("Sizes of uniform recipes are exact:")
    for name, row in rows.items():
        if not name.startswith("Mixed"):
            check(row["bytes"] == old[name]["bytes"], f"{name}: {row['bytes']:,} bytes")

    print("Qualitative claims:")
    kernel = fresh["benchmark"]["integer_kernel"]
    check(kernel["top1_agreement_with_simulation"] == 1.0, "integer kernel agrees with simulation on every test image")
    check(kernel["max_code_difference_lsb"] <= 1, "integer kernel codes within 1 LSB of simulation")
    check(
        rows["W4A8 per-channel, min-max"]["kl"]["mean"] < rows["W4A8 per-tensor, min-max"]["kl"]["mean"],
        "per-channel scales lower output KL at 4 bits",
    )
    check(rows["W2A8 per-channel, min-max"]["accuracy"]["mean"] < 0.8, "2-bit weights collapse")
    greedy, byte_aware = (
        rows["Mixed precision, searched (0.5% budget)"],
        rows["Mixed precision, searched, KL-per-byte order (0.5% budget)"],
    )
    check(byte_aware["bytes"] <= greedy["bytes"], "KL-per-byte search is no larger than KL-order search")
    check(
        byte_aware["kl"]["mean"] < min(rows[n]["kl"]["mean"] for n in rows if n.startswith("W4")),
        "KL-per-byte search has lower KL than every uniform 4-bit recipe",
    )
    matches = fresh["search_ablation"]["matches_exhaustive"]
    n = len(fresh["search_ablation"]["seeds"])
    hits = matches["kl_per_byte"]
    check(hits == n, f"KL-per-byte greedy matches the exhaustive optimum on {hits}/{n}")
    check(matches["kl"] < n, f"KL-order greedy misses the optimum on at least one seed ({matches['kl']}/{n})")

    if FAILURES:
        print(f"\n{len(FAILURES)} claim(s) did not hold.")
        return 1
    print("\nAll claims hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/demo"))
