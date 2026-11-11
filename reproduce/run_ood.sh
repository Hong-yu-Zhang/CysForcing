#!/usr/bin/env bash
# CysForcing predictions on the Mouse and SL1344 chemoproteomic sets.
#
#   bash reproduce/run_ood.sh
#
# Expects the structures unpacked at data/ood/af2_pdb (see README "Data
# preparation"); they are fetched from AlphaFold DB if that folder is empty.
# Override any of WEIGHT / CONFIG / PDB_FOLDER / OUT_DIR via the environment.
#
# The per-site predictions written to $OUT_DIR/pred_exp_inter_with_prob.tsv
# are what paper/fig4.py scores under the paper's labelling convention. Step 4
# is a single-cutoff threshold sweep kept as a diagnostic only.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WEIGHT="${WEIGHT:-weights/CysForcing.pth}"
CONFIG="${CONFIG:-configs/model/CysForcing.yaml}"
TRUTH="${TRUTH:-data/ood/pred_exp_inter.tsv}"
PDB_FOLDER="${PDB_FOLDER:-data/ood/af2_pdb}"
OUT_DIR="${OUT_DIR:-result_inference/ood}"

mkdir -p "$OUT_DIR"

if [ ! -d "$PDB_FOLDER" ] || [ -z "$(ls -A "$PDB_FOLDER" 2>/dev/null)" ]; then
    echo "[1/4] downloading AlphaFold structures for $TRUTH ..."
    python scripts/get_AF2.py -i "$TRUTH" -o "$PDB_FOLDER" -j 12
else
    echo "[1/4] structures present: $(ls "$PDB_FOLDER" | wc -l) files"
fi

echo "[2/4] inference ..."
python simple_inference.py \
    --model "$CONFIG" --weight "$WEIGHT" \
    --pdb_folder "$PDB_FOLDER" --out_dir "$OUT_DIR" --out_name result.json

echo "[3/4] joining predictions with measured ratios ..."
python scripts/add_prob.py \
    -i "$TRUTH" -j "$OUT_DIR/result.json" \
    -o "$OUT_DIR/pred_exp_inter_with_prob.tsv" --with-status

echo "[4/4] diagnostic sweep over single ratio cutoffs ..."
python scripts/calc_metric.py \
    -i "$OUT_DIR/pred_exp_inter_with_prob.tsv" \
    --species-thresholds-json '{"SL1344":[1,2,3,4,5,6,7],"Mouse":[1,2,3,4,5,6,7]}' \
    --prob-thr 0.5 --require-ok \
    -o "$OUT_DIR/metric_plots"

echo
echo "predictions -> $OUT_DIR/pred_exp_inter_with_prob.tsv"
echo "Fig. 4 metrics: python paper/fig4_metrics.py --cysforcing-ood $OUT_DIR/pred_exp_inter_with_prob.tsv"
