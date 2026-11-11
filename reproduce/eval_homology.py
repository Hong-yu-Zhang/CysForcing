"""Evaluate CysForcing on the held-out test set at four homology cutoffs.

The test split is restricted to proteins whose highest sequence identity to any
training protein is at most 100 / 80 / 60 / 40 %, and the model is scored on
each nested subset. The 100 % column is the unfiltered test set.

    python reproduce/eval_homology.py
"""
import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs import get_cfg_defaults
from dataloader import CysDataset
from models import CysForcing
from utils import pad_collate, set_seed

CUTOFFS = (100.0, 80.0, 60.0, 40.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="configs/model/CysForcing.yaml")
    p.add_argument("--data", default="configs/data/cys.yaml")
    p.add_argument("--weight", default="weights/CysForcing.pth")
    p.add_argument("--feat_root", default=None,
                   help="override DATA.FEAT_ROOT from the data config")
    p.add_argument("--out", default="reproduce/out",
                   help="directory for the metric table")
    return p.parse_args()


def score(y_true, y_score):
    # The classifier output is thresholded at 0.5 before the sigmoid, i.e. at
    # probability 0.622. Every published binary metric -- for this model and
    # for the baselines it is compared against -- uses that convention, so it
    # is kept verbatim. AUROC and AUPRC are threshold-free.
    y_hat = (np.asarray(y_score) > 0.5).astype(int)
    return dict(
        auroc=roc_auc_score(y_true, y_score),
        auprc=average_precision_score(y_true, y_score),
        acc=accuracy_score(y_true, y_hat),
        balanced_acc=balanced_accuracy_score(y_true, y_hat),
        mcc=matthews_corrcoef(y_true, y_hat),
        precision=precision_score(y_true, y_hat, zero_division=0),
        recall=recall_score(y_true, y_hat),
        f1=f1_score(y_true, y_hat),
    )


def predict(model, loader, device):
    """Collect classifier outputs at the Cys sites that carry a measurement."""
    labels, scores = [], []
    model.eval()
    with torch.no_grad():
        for esm, saprot, y, ratios, _cys_mask in loader:
            esm, saprot = esm.to(device), saprot.to(device)
            y = y.float().to(device)
            ratios = ratios.float().to(device)
            # The last head is the fused classifier; earlier entries are the
            # per-modality auxiliary heads used only during training.
            out = model(esm, saprot)["binary"][-1].squeeze(-1)
            mask = ratios.bool()
            scores.extend(torch.masked_select(out, mask).cpu().tolist())
            labels.extend(torch.masked_select(y, mask).cpu().tolist())
    return np.asarray(labels), np.asarray(scores)


def main():
    args = parse_args()
    os.chdir(ROOT)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.model)
    cfg.merge_from_file(args.data)
    set_seed(cfg.SOLVER.SEED)
    feat_root = args.feat_root or cfg.DATA.FEAT_ROOT

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CysForcing(**cfg)
    model.load_state_dict(torch.load(args.weight, map_location="cpu"))
    model.to(device)

    df_all = pd.read_csv(os.path.join(cfg.DATA.SPLIT_DIR, "test.csv"))
    df_all["max_identity"] = df_all["max_identity"].astype(float)

    rows = []
    for high in CUTOFFS:
        df = df_all[df_all["max_identity"] <= high].reset_index(drop=True)
        loader = DataLoader(
            CysDataset(df, feat_root),
            batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
            num_workers=cfg.SOLVER.NUM_WORKERS, drop_last=False,
            collate_fn=pad_collate)
        y, p = predict(model, loader, device)
        # ESM-2 truncates at 1024 tokens, so Cys sites past that offset carry a
        # label but never reach the classifier. They are dropped here exactly
        # as they were when the published numbers were produced.
        annotated = sum(sum(1 for v in ast.literal_eval(r) if v != 0)
                        for r in df["Label_reg"])
        rows.append(dict(cutoff=high, proteins=len(df), annotated_sites=annotated,
                         scored_sites=len(y), positives=int(y.sum()),
                         **score(y, p)))
        print(f"  <= {high:>5.0f}%  {len(df):>3} proteins  "
              f"{len(y):>3}/{annotated} sites scored  "
              f"balanced_acc {rows[-1]['balanced_acc']:.4f}  "
              f"mcc {rows[-1]['mcc']:.4f}  f1 {rows[-1]['f1']:.4f}  "
              f"recall {rows[-1]['recall']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "homology_metrics.tsv")
    pd.DataFrame(rows).to_csv(dest, sep="\t", index=False)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
