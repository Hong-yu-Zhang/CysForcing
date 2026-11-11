#!/usr/bin/env bash
# Train and score CysForcing with an alternative sequence backbone, from cached features.
#
#   bash reproduce/run_backbone.sh esmc300m
#
# Expects configs/model/CysForcing_<tag>.yaml, configs/data/cys_<tag>.yaml and the
# sequence features at data/features_<tag>/{train,ood,ciaa}/esm_feat (see
# scripts/build_esmc_features.py). SaProt features are shared with the ESM-2 runs:
# data/features, data/features_ood and data/features_ciaa, built by
# scripts/build_features.py on data/raw/AF2_pdb, data/ood/af2_pdb and
# data/ciaa/af2_denovo.
set -euo pipefail

TAG="$1"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="configs/model/CysForcing_${TAG}.yaml"
DATA="configs/data/cys_${TAG}.yaml"
FEAT="data/features_${TAG}"
OUT="result_inference/${TAG}"
CIAA_LABELS="${CIAA_LABELS:-data/ciaa/af2_denovo_labels.csv}"

for pair in train:features ood:features_ood ciaa:features_ciaa; do
    ln -sfn "$ROOT/data/${pair#*:}/saprot_feat" "$FEAT/${pair%%:*}/saprot_feat"
done

echo "[1/4] training ..."
python main.py --model "$MODEL" --data "$DATA"
EXP="$(ls -d result/CysForcing_${TAG}/exp* | sort -V | tail -1)"
WEIGHT="$EXP/best_model_epoch.pth"
echo "weights: $WEIGHT"

echo "[2/4] test set at four homology cutoffs ..."
python reproduce/eval_homology.py --model "$MODEL" --data "$DATA" \
    --weight "$WEIGHT" --out "reproduce/out/$TAG"

echo "[3/4] Mouse + SL1344 ..."
python reproduce/infer_cached.py --model "$MODEL" --weight "$WEIGHT" \
    --feat_root "$FEAT/ood" --pdb_folder data/ood/af2_pdb --out "$OUT/ood/result.json"
python scripts/add_prob.py -i data/ood/pred_exp_inter.tsv -j "$OUT/ood/result.json" \
    -o "$OUT/ood/pred_exp_inter_with_prob.tsv" --with-status
python scripts/calc_metric.py -i "$OUT/ood/pred_exp_inter_with_prob.tsv" \
    --species-thresholds-json '{"SL1344":[1,2,3,4,5,6,7],"Mouse":[1,2,3,4,5,6,7]}' \
    --prob-thr 0.5 --require-ok -o "$OUT/ood/metric_plots"

echo "[4/4] CIAA AF2-denovo ..."
python reproduce/infer_cached.py --model "$MODEL" --weight "$WEIGHT" \
    --feat_root "$FEAT/ciaa" --pdb_folder data/ciaa/af2_denovo --out "$OUT/ciaa_denovo/result.json"
python reproduce/eval_ciaa_denovo.py --pred_json "$OUT/ciaa_denovo/result.json" \
    --labels "$CIAA_LABELS" --out "$OUT/ciaa_denovo"

echo
echo "done -> $EXP, reproduce/out/$TAG, $OUT"
