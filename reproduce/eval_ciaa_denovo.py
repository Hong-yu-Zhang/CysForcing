"""Score CysForcing on the CIAA AF2-denovo set.

The labels come from the CIAA release (isotop_alphafold_denovo_set.csv); the
columns this script reads ship as data/ciaa/af2_denovo_labels.csv, one row per
cysteine with an `identifier` such as "O00329_C147", its `ratio_mean` and a
`reactivity` label. Rows labelled 0.5 are ambiguous and are dropped, matching
how the set is scored in the CIAA paper.

Sites that CIAA annotates but this model never scored -- the structure was not
downloaded, or the position falls past the 1024-token truncation -- are reported
separately rather than silently counted as errors.

Usage:
    python reproduce/eval_ciaa_denovo.py \
        --pred_json result_inference/ciaa_denovo/result.json \
        --labels    data/ciaa/af2_denovo_labels.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pred_json", required=True,
                   help="result.json written by simple_inference.py")
    p.add_argument("--labels", required=True,
                   help="CIAA labels, e.g. data/ciaa/af2_denovo_labels.csv")
    p.add_argument("--out", default="reproduce/out",
                   help="directory for the metric table")
    p.add_argument("--prob_thr", type=float, default=0.5,
                   help="probability above which a site is called reactive")
    return p.parse_args()


def to_int_label(v):
    if isinstance(v, str):
        return 1 if v.strip().lower() == "high" else 0
    return int(float(v))


def load_prob_index(path):
    """{(uniprot, 1-indexed Cys position): probability}."""
    with open(path) as fh:
        data = json.load(fh)
    idx = {}
    for item in data:
        for pdb_name, sites in item.items():
            if not pdb_name.endswith(".pdb"):
                continue
            uniprot = pdb_name[:-4]
            for key, val in sites.items():
                if key == "Sequence" or not key.startswith("C_"):
                    continue
                try:
                    pos = int(key.split("_", 1)[1])
                except ValueError:
                    continue
                if isinstance(val, list) and val:
                    idx[(uniprot, pos)] = float(val[0])
                elif isinstance(val, (int, float)):
                    idx[(uniprot, pos)] = float(val)
    return idx


def parse_identifier(identifier):
    parts = str(identifier).split("_")
    if len(parts) < 2 or not parts[1].startswith("C"):
        return None, None
    try:
        return parts[0], int(parts[1][1:])
    except ValueError:
        return None, None


def main():
    args = parse_args()
    df = pd.read_csv(args.labels, usecols=["identifier", "ratio_mean", "reactivity"])
    df = df[df["reactivity"] != 0.5].copy()
    df["y_true"] = df["reactivity"].map(to_int_label)
    df[["uniprot", "pos"]] = df["identifier"].map(parse_identifier).apply(pd.Series)

    prob_idx = load_prob_index(args.pred_json)
    scored_uniprots = {u for u, _ in prob_idx}

    def lookup(row):
        u, p = row["uniprot"], row["pos"]
        if pd.isna(u) or pd.isna(p):
            return pd.Series({"prob": np.nan, "status": "bad_identifier"})
        key = (u, int(p))
        if key in prob_idx:
            return pd.Series({"prob": prob_idx[key], "status": "ok"})
        if u not in scored_uniprots:
            return pd.Series({"prob": np.nan, "status": "structure_missing"})
        return pd.Series({"prob": np.nan, "status": "position_not_scored"})

    df[["prob", "status"]] = df.apply(lookup, axis=1)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "ciaa_denovo_per_site.tsv"),
              sep="\t", index=False, float_format="%.6f")

    print(f"annotated sites: {len(df)}")
    for status, count in df["status"].value_counts().items():
        print(f"  {status}: {count}")

    ok = df[df["status"] == "ok"]
    y_true = ok["y_true"].astype(int).values
    y_prob = ok["prob"].astype(float).values
    if len(set(y_true.tolist())) < 2:
        raise SystemExit("scored sites are single-class; cannot compute metrics")
    y_hat = (y_prob >= args.prob_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()

    res = dict(dataset="CIAA_AF2_denovo", n=len(ok),
               positives=int(y_true.sum()), negatives=int((1 - y_true).sum()),
               accuracy=accuracy_score(y_true, y_hat),
               precision=precision_score(y_true, y_hat, zero_division=0),
               recall=recall_score(y_true, y_hat, zero_division=0),
               f1=f1_score(y_true, y_hat, zero_division=0),
               mcc=matthews_corrcoef(y_true, y_hat),
               roc_auc=roc_auc_score(y_true, y_prob),
               TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))

    dest = os.path.join(args.out, "ciaa_denovo_metrics.tsv")
    pd.DataFrame([res]).to_csv(dest, sep="\t", index=False)
    print(f"\nscored {res['n']} sites "
          f"({res['positives']} reactive, {res['negatives']} unreactive)")
    print(f"  f1  {res['f1']:.4f}   mcc {res['mcc']:.4f}   auc {res['roc_auc']:.4f}")
    print(f"  precision {res['precision']:.4f}   recall {res['recall']:.4f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
