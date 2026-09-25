#!/usr/bin/env bash
# One-command quick start from a clean checkout. Needs python3 >= 3.10; no GPU, no downloads
# beyond the pip wheels (the digits set ships with scikit-learn).
#
# Creates .venv, checks the dataset checksum, reruns the 5-seed benchmark, the bit-width sweep
# and the search ablation, and compares every number with the committed results/. Fresh outputs,
# the environment and this log are written to runs/demo/.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/demo
mkdir -p "$OUT"
exec > >(tee "$OUT/demo.log") 2>&1

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e .

python - <<'PY' | tee "runs/demo/environment.txt"
import hashlib, platform, subprocess, sys
from importlib.metadata import version

import numpy as np
from sklearn.datasets import load_digits

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print("commit:", commit or "unknown")
print("python:", sys.version.split()[0], "on", platform.platform())
print("numpy:", version("numpy"), " scikit-learn:", version("scikit-learn"))
digits = load_digits()
h = hashlib.sha256(np.ascontiguousarray(digits.data, dtype=np.float64).tobytes())
h.update(np.ascontiguousarray(digits.target, dtype=np.int64).tobytes())
expected = "f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443"
print("digits sha256:", h.hexdigest(), "(expected)" if h.hexdigest() == expected else "(UNEXPECTED)")
sys.exit(h.hexdigest() != expected)
PY

echo; echo "== benchmark: 12 recipes x 5 seeds"
edge-quant benchmark --seeds 5 --out "$OUT/benchmark.json"
echo; echo "== bit-width sweep"
edge-quant sweep --seeds 5 --out "$OUT/bit_sweep.json" > /dev/null
echo; echo "== greedy search vs exhaustive optimum"
edge-quant ablate-search --seeds 5 --out "$OUT/search_ablation.json"

echo; echo "== fresh results vs committed results/"
for name in benchmark bit_sweep search_ablation; do
  python scripts/compare_results.py "$OUT/$name.json" "results/$name.json"
done
echo; echo "Done. Outputs are in $OUT/."
