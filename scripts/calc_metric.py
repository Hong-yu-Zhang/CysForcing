#!/usr/bin/env python3
"""Score predictions per species across a sweep of ratio thresholds.

The input TSV must carry the columns `ratio`, `species` and `prob`, as produced
by scripts/add_prob.py. A site is labelled reactive when its measured depletion
ratio falls below the threshold, and predicted reactive when its probability
reaches `--prob-thr`:

    y_true = 1 if ratio < ratio_threshold else 0
    y_pred = 1 if prob  >= prob_threshold  else 0

One plot per species is written, sweeping F1, Recall, Precision and MCC over the
thresholds, alongside a table with one row per species and threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score


# Species that may appear in the input, with the ratio threshold used when no
# sweep is requested. Species outside this table are skipped.
RATIO_THRESHOLD: Dict[str, float] = {
    "SL1344": 5.0,
    "Mouse": 0.5,
}


def parse_species_thresholds_json(text: str) -> Dict[str, List[float]]:
    """Parse `{"SL1344": [3, 4, 5], "Mouse": [0.3, 0.5, 0.7]}`."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--species-thresholds-json is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise SystemExit("--species-thresholds-json must be a JSON object keyed by species.")

    parsed: Dict[str, List[float]] = {}
    for species, values in raw.items():
        if not isinstance(species, str):
            raise SystemExit("species-thresholds keys must be species names.")
        if not isinstance(values, list) or not values:
            raise SystemExit(f"threshold list for {species} is empty or malformed.")
        try:
            parsed[species] = sorted(float(v) for v in values)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"thresholds for {species} must all be numeric.") from exc

    return parsed


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-i", "--tsv", required=True, help="input TSV")
    parser.add_argument(
        "--species-thresholds-json",
        default="",
        help="ratio thresholds to sweep per species, e.g. "
             "'{\"SL1344\":[3,4,5],\"Mouse\":[0.3,0.5,0.7]}'",
    )
    parser.add_argument("--prob-thr", type=float, default=0.5,
                        help="probability above which a site is called reactive")
    parser.add_argument("--require-ok", action="store_true",
                        help="keep only rows with prob_status=ok, when that column exists")
    parser.add_argument("--unknown", choices=["skip", "error"], default="skip",
                        help="what to do with a species outside RATIO_THRESHOLD")
    parser.add_argument("-o", "--out-dir", default="metric_plots", help="output directory")
    args = parser.parse_args()

    if args.species_thresholds_json.strip():
        species_thresholds = parse_species_thresholds_json(args.species_thresholds_json)
    else:
        species_thresholds = {sp: [thr] for sp, thr in RATIO_THRESHOLD.items()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_by_species: Dict[str, List[float]] = defaultdict(list)
    prob_by_species: Dict[str, List[float]] = defaultdict(list)

    with open(args.tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise SystemExit("input TSV has no header.")

        missing = {"ratio", "species", "prob"} - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"input TSV is missing required columns: {sorted(missing)}")

        has_prob_status = "prob_status" in reader.fieldnames

        for row in reader:
            species = (row.get("species") or "").strip()
            if not species:
                continue

            if species not in RATIO_THRESHOLD:
                if args.unknown == "error":
                    raise SystemExit(f"species {species} is not in RATIO_THRESHOLD.")
                continue

            if args.require_ok and has_prob_status:
                if (row.get("prob_status") or "").strip().lower() != "ok":
                    continue

            try:
                ratio = float((row.get("ratio") or "").strip())
                prob = float((row.get("prob") or "").strip())
            except ValueError:
                continue

            ratio_by_species[species].append(ratio)
            prob_by_species[species].append(prob)

    metric_tsv_path = out_dir / "species_threshold_metrics.tsv"
    with open(metric_tsv_path, "w", encoding="utf-8", newline="") as wf:
        writer = csv.writer(wf, delimiter="\t")
        writer.writerow(["species", "ratio_threshold", "prob_threshold", "n",
                         "positive", "negative", "precision", "recall", "f1", "mcc"])

        plotted_species = 0
        for species in sorted(species_thresholds.keys()):
            if not ratio_by_species.get(species):
                print(f"[WARN] {species}: no usable rows in the input, skipping.")
                continue

            ratio_arr = np.asarray(ratio_by_species[species], dtype=float)
            prob_arr = np.asarray(prob_by_species[species], dtype=float)
            y_pred_arr = (prob_arr >= args.prob_thr).astype(int)
            n = int(ratio_arr.size)

            x_thr, f1_list, recall_list, precision_list, mcc_list = [], [], [], [], []

            for thr in species_thresholds[species]:
                y_true_arr = (ratio_arr < thr).astype(int)
                pos = int((y_true_arr == 1).sum())
                neg = int((y_true_arr == 0).sum())
                precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
                recall = recall_score(y_true_arr, y_pred_arr, zero_division=0)
                f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
                mcc = matthews_corrcoef(y_true_arr, y_pred_arr)

                x_thr.append(thr)
                f1_list.append(float(f1))
                recall_list.append(float(recall))
                precision_list.append(float(precision))
                mcc_list.append(float(mcc))

                writer.writerow([species, thr, args.prob_thr, n, pos, neg,
                                 f"{precision:.6f}", f"{recall:.6f}",
                                 f"{f1:.6f}", f"{mcc:.6f}"])

            plt.figure(figsize=(8, 5))
            plt.plot(x_thr, f1_list, marker="o", label="F1")
            plt.plot(x_thr, recall_list, marker="o", label="Recall")
            plt.plot(x_thr, precision_list, marker="o", label="Precision")
            plt.plot(x_thr, mcc_list, marker="o", label="MCC")
            plt.xlabel("Ratio Threshold")
            plt.ylabel("Metric Value")
            plt.title(f"{species} Metrics vs Ratio Threshold (prob_thr={args.prob_thr})")
            plt.ylim(-1.0, 1.05)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()

            png_path = out_dir / f"{sanitize_name(species)}_metrics.png"
            plt.savefig(png_path, dpi=200)
            plt.close()
            plotted_species += 1
            print(f"[OK] {species}: plot -> {png_path}")

    print(f"[DONE] plots: {plotted_species}")
    print(f"[DONE] metric table: {metric_tsv_path}")


if __name__ == "__main__":
    main()
