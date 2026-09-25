"""Compare a freshly generated results JSON with the committed one.

    python scripts/compare_results.py FRESH COMMITTED [--ignore KEY ...] [--rtol R] [--atol A]

Numbers must agree within the tolerances; strings, booleans and structure must match exactly.
Keys passed with --ignore (for example wall-clock timings) are skipped wherever they appear.
Exits 1 and lists every disagreeing field.
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def compare(fresh, committed, *, ignore=(), rtol=1e-9, atol=1e-12):
    """Return (number of fields checked, list of differences)."""
    diffs: list[str] = []
    checked = 0

    def walk(a, b, path):
        nonlocal checked
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key in ignore:
                    continue
                if key not in a or key not in b:
                    diffs.append(f"{path}/{key}: present on only one side")
                    continue
                walk(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                diffs.append(f"{path}: length {len(a)} vs {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, f"{path}[{i}]")
        elif _is_number(a) and _is_number(b):
            checked += 1
            if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                diffs.append(f"{path}: fresh {a!r} vs committed {b!r}")
        else:
            checked += 1
            if a != b:
                diffs.append(f"{path}: fresh {a!r} vs committed {b!r}")

    walk(fresh, committed, "")
    return checked, diffs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fresh")
    parser.add_argument("committed")
    parser.add_argument("--ignore", nargs="*", default=[], help="keys to skip, e.g. timing fields")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args(argv)
    with open(args.fresh) as f:
        fresh = json.load(f)
    with open(args.committed) as f:
        committed = json.load(f)
    checked, diffs = compare(fresh, committed, ignore=set(args.ignore), rtol=args.rtol, atol=args.atol)
    if diffs:
        print(f"{args.fresh} differs from {args.committed} in {len(diffs)} of {checked} fields:")
        for line in diffs[:50]:
            print("  " + line)
        return 1
    skipped = f" (ignoring {', '.join(args.ignore)})" if args.ignore else ""
    print(f"OK: {args.fresh} matches {args.committed} on all {checked} fields{skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
