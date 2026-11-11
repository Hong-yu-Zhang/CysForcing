#!/usr/bin/env bash
# Regenerate every data-driven figure of the paper from paper/data.
#
#   bash paper/run_all.sh            # -> paper/out/
#
# Needs only the Python requirements; no GPU, structures or feature cache.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

python paper/fig2.py
python paper/fig3.py
python paper/fig4_metrics.py
python paper/fig4.py
python paper/figS1.py

echo
echo "done -> paper/out/"
