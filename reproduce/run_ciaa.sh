#!/usr/bin/env bash
# CysForcing predictions on the CIAA AF2-denovo set.
#
#   bash reproduce/run_ciaa.sh
#
# Expects the structures unpacked at data/ciaa/af2_denovo (see README "Data
# preparation"); they are fetched from AlphaFold DB if that folder is empty.
# Labels default to data/ciaa/af2_denovo_labels.csv, taken from the CIAA
# release (see THIRD_PARTY_NOTICES.md); override with CIAA_LABELS.
#
# The per-site predictions written to $OUT_DIR/ciaa_denovo_per_site.tsv are
# what paper/fig4.py scores alongside the three released predictors.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WEIGHT="${WEIGHT:-weights/CysForcing.pth}"
CONFIG="${CONFIG:-configs/model/CysForcing.yaml}"
UNIPROTS="${UNIPROTS:-data/ciaa/af2_denovo_val.uniprots.txt}"
PDB_FOLDER="${PDB_FOLDER:-data/ciaa/af2_denovo}"
OUT_DIR="${OUT_DIR:-result_inference/ciaa_denovo}"
CIAA_LABELS="${CIAA_LABELS:-data/ciaa/af2_denovo_labels.csv}"

if [ ! -f "$CIAA_LABELS" ]; then
    echo "CIAA label file not found: $CIAA_LABELS" >&2
    echo "Download the dataset archive from Zenodo into data/ first (see data/README.md)," >&2
    echo "or point CIAA_LABELS at your own copy." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

if [ ! -d "$PDB_FOLDER" ] || [ -z "$(ls -A "$PDB_FOLDER" 2>/dev/null)" ]; then
    echo "[1/3] downloading AlphaFold structures for $(wc -l < "$UNIPROTS") UniProts ..."
    python scripts/get_AF2.py -i "$UNIPROTS" -o "$PDB_FOLDER" -j 12
else
    echo "[1/3] structures present: $(ls "$PDB_FOLDER" | wc -l) files"
fi

echo "[2/3] inference ..."
python simple_inference.py \
    --model "$CONFIG" --weight "$WEIGHT" \
    --pdb_folder "$PDB_FOLDER" --out_dir "$OUT_DIR" --out_name result.json

echo "[3/3] scoring against the CIAA labels ..."
python reproduce/eval_ciaa_denovo.py \
    --pred_json "$OUT_DIR/result.json" \
    --labels "$CIAA_LABELS" \
    --out "$OUT_DIR"

echo
echo "done -> $OUT_DIR/ciaa_denovo_metrics.tsv"
