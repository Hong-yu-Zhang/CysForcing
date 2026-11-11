"""Fig. 4 metrics: CysForcing against three released predictors on the OOD sets.

Reads the per-site source data (paper/data/ood_sites.tsv), labels each site with
the hyper-reactivity convention of the compared predictors (CIAA, HyperCys,
sbPCR) and writes three tables to paper/out/:

  fig4_radar.tsv      per set and method: AUROC, AUPRC, MCC, balanced accuracy,
                      plus precision / recall / F1 (Fig. 4a-c)
  fig4_homology.tsv   the same sites split by whether homology transfer has a
                      measured training counterpart to copy (Fig. 4d-i)
  fig4_bootstrap.tsv  protein-grouped paired bootstrap on AUROC

    python paper/fig4_metrics.py

To score freshly generated CysForcing predictions instead of the shipped ones,
pass the files written by reproduce/run_ood.sh and reproduce/run_ciaa.sh:

    python paper/fig4_metrics.py \
        --cysforcing-ood  result_inference/ood/pred_exp_inter_with_prob.tsv \
        --cysforcing-ciaa result_inference/ciaa_denovo/ciaa_denovo_per_site.tsv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))

POS_MAX = 2.3     # hyper-reactive: ratio <= POS_MAX
NEG_MIN = 10.0    # low-reactive:   ratio >= NEG_MIN
SETS = ["Mouse", "SL1344", "CIAA"]
# (score column, display name, native-label column or None -> score >= 0.5)
METHODS = [("cysforcing", "CysForcing", None), ("ciaa_rf", "CIAA RF", None),
           ("hypercys", "HyperCys", None), ("sbpcr", "sbPCR", "sbpcr_pred")]
HOMOLOGY = "Homology transfer"
N_DRAW = 100          # random draws for homology transfer where it has nothing to copy
RESAMPLES = 600       # bootstrap resamples
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sites", default=os.path.join(HERE, "data", "ood_sites.tsv"))
    p.add_argument("--out", default=os.path.join(HERE, "out"))
    p.add_argument("--cysforcing-ood", default=None,
                   help="pred_exp_inter_with_prob.tsv from reproduce/run_ood.sh")
    p.add_argument("--cysforcing-ciaa", default=None,
                   help="ciaa_denovo_per_site.tsv from reproduce/run_ciaa.sh")
    p.add_argument("--no-bootstrap", action="store_true")
    return p.parse_args()


def override_cysforcing(sites, ood_path, ciaa_path):
    """replace the shipped CysForcing scores by freshly generated ones"""
    sites = sites.copy()
    if ood_path:
        d = pd.read_csv(ood_path, sep="\t")
        d = d[d["prob_status"] == "ok"]
        prob = {(r.species, r.seq_id, int(r.posi)): float(r.prob) for r in d.itertuples()}
        ood = sites["set"].isin(["Mouse", "SL1344"])
        sites.loc[ood, "cysforcing"] = [
            prob.get((s, u, p), np.nan)
            for s, u, p in zip(sites.loc[ood, "set"], sites.loc[ood, "uniprot"], sites.loc[ood, "pos0"])]
    if ciaa_path:
        d = pd.read_csv(ciaa_path, sep="\t")
        d = d[d["status"] == "ok"]
        prob = {(r.uniprot, int(r.pos) - 1): float(r.prob) for r in d.itertuples()}
        ciaa = sites["set"] == "CIAA"
        sites.loc[ciaa, "cysforcing"] = [
            prob.get((u, p), np.nan)
            for u, p in zip(sites.loc[ciaa, "uniprot"], sites.loc[ciaa, "pos0"])]
    missing = int(sites["cysforcing"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} sites have no CysForcing prediction in the given files")
    return sites


def label(sites):
    """keep sites in either band of the convention and attach the binary label"""
    r = sites["ratio"].astype(float)
    out = sites[(r <= POS_MAX) | (r >= NEG_MIN)].copy()
    out["y"] = (out["ratio"].astype(float) <= POS_MAX).astype(int)
    return out.reset_index(drop=True)


def native_pred(d, score_col, pred_col):
    if pred_col:
        return d[pred_col].to_numpy(int)
    return (d[score_col].to_numpy(float) >= 0.5).astype(int)


def two_class(y):
    return len(np.unique(y)) > 1


def radar_table(lab):
    rows = []
    for ds in SETS:
        d = lab[lab["set"] == ds]
        y = d["y"].to_numpy(int)
        for col, name, pred_col in METHODS:
            s = d[col].to_numpy(float)
            pred = native_pred(d, col, pred_col)
            rows.append(dict(
                dataset=ds, method=name, n=len(d), positives=int(y.sum()),
                auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s),
                mcc=matthews_corrcoef(y, pred), bacc=balanced_accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0)))
    return pd.DataFrame(rows)


def regime_metrics(y, score, pred):
    out = {"auroc": np.nan, "recall": np.nan, "f1": np.nan, "mcc": np.nan}
    if two_class(y):
        out["auroc"] = roc_auc_score(y, score)
        out["mcc"] = matthews_corrcoef(y, pred)
    out["recall"] = recall_score(y, pred, zero_division=0)
    out["f1"] = f1_score(y, pred, zero_division=0)
    return out


def homology_table(lab):
    rows = []
    for ds in SETS:
        df = lab[lab["set"] == ds]
        for regime, mask in [("transferable", df["transferable"].astype(bool)),
                             ("non-transferable", ~df["transferable"].astype(bool))]:
            d = df[mask]
            y = d["y"].to_numpy(int)
            n, pos = len(d), int(y.sum())
            for col, name, pred_col in METHODS:
                m = regime_metrics(y, d[col].to_numpy(float), native_pred(d, col, pred_col))
                rows.append(dict(dataset=ds, regime=regime, n=n, positives=pos, method=name, **m))
            if regime == "transferable":
                # copy the aligned training cysteine's measured ratio
                t = d["transferred_ratio"].to_numpy(float)
                m = regime_metrics(y, -np.log2(np.clip(t, 1e-6, None)), (t <= POS_MAX).astype(int))
            else:
                # nothing to copy: a random predictor at the observed positive rate
                rng = np.random.default_rng(SEED)
                p = y.mean()
                draws = []
                for _ in range(N_DRAW):
                    s = rng.normal(size=n)
                    pred = (rng.random(n) < p).astype(int)
                    draws.append(regime_metrics(y, s, pred))
                m = {k: float(np.nanmean([a[k] for a in draws])) for k in draws[0]}
            rows.append(dict(dataset=ds, regime=regime, n=n, positives=pos, method=HOMOLOGY, **m))
    return pd.DataFrame(rows)


def paired_bootstrap(y, scores, group, reference, resamples=RESAMPLES, seed=SEED):
    """protein-grouped paired bootstrap of AUROC; gap > 0 means better than reference"""
    index = {}
    for i, g in enumerate(group):
        index.setdefault(g, []).append(i)
    labels = np.array(sorted(index))
    blocks = [np.asarray(index[g]) for g in labels]
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in scores}
    for _ in range(resamples):
        rows = np.concatenate([blocks[i] for i in rng.integers(0, len(labels), len(labels))])
        t = y[rows]
        if np.ptp(t) == 0:
            continue
        for k, s in scores.items():
            draws[k].append(roc_auc_score(t, s[rows]))
    base = np.asarray(draws[reference], float)
    out = {}
    for k in scores:
        if k == reference:
            continue
        gap = np.asarray(draws[k], float) - base
        out[k] = dict(gap=float(gap.mean()), lo=float(np.percentile(gap, 2.5)),
                      hi=float(np.percentile(gap, 97.5)), p_better=float((gap > 0).mean()))
    return out


def bootstrap_table(lab):
    names = {col: name for col, name, _ in METHODS}
    baselines = [col for col, _, _ in METHODS[1:]]
    rows = []
    for ds in SETS:
        d = lab[lab["set"] == ds]
        y = d["y"].to_numpy(float)
        g = d["uniprot"].to_numpy()
        sc = {col: d[col].to_numpy(float) for col in names}
        res = paired_bootstrap(y, sc, g, reference="cysforcing")
        for col in baselines:
            rows.append(dict(dataset=ds, comparison=f"{names[col]} - CysForcing", **res[col]))
        for a, b in itertools.combinations(baselines, 2):
            e = paired_bootstrap(y, {a: sc[a], b: sc[b]}, g, reference=a)[b]
            rows.append(dict(dataset=ds, comparison=f"{names[b]} - {names[a]}", **e))
    return pd.DataFrame(rows)


def summary(radar, homology, boot):
    others = [name for _, name, _ in METHODS[1:]]
    print("Fig. 4a-c  (labels: ratio <= %g hyper-reactive, ratio >= %g low-reactive)" % (POS_MAX, NEG_MIN))
    for ds in SETS:
        s = radar[radar.dataset == ds].set_index("method")
        cf = s.loc["CysForcing"]
        lead = {k: cf[k] - s.loc[others, k].max() for k in ("auroc", "auprc", "mcc", "bacc")}
        print(f"  {ds:<7} n={int(cf.n):<4} AUROC {cf.auroc:.3f} (+{lead['auroc']:.3f})  "
              f"AUPRC +{lead['auprc']:.3f}  MCC +{lead['mcc']:.3f}  balACC +{lead['bacc']:.3f}  "
              f"| released predictors AUROC {s.loc[others, 'auroc'].min():.3f}-{s.loc[others, 'auroc'].max():.3f}")
    print("Fig. 4d-i  (non-transferable = homology transfer has no measured counterpart)")
    for ds in SETS:
        h = homology[homology.dataset == ds]
        nt = h[(h.regime == "non-transferable") & (h.method == "CysForcing")].iloc[0]
        tot = h[h.method == "CysForcing"]["n"].sum()
        ht = h[(h.regime == "non-transferable") & (h.method == HOMOLOGY)].iloc[0]
        print(f"  {ds:<7} non-transferable {int(nt.n)}/{int(tot)} ({nt.n / tot * 100:.0f}%)  "
              f"CysForcing AUROC {nt.auroc:.3f}  homology transfer {ht.auroc:.3f}")
    if boot is not None:
        vs = boot[boot.comparison.str.endswith("- CysForcing")]
        pair = boot[~boot.comparison.str.endswith("- CysForcing")]
        print("Bootstrap  (AUROC, protein-grouped, %d resamples)" % RESAMPLES)
        print(f"  every released predictor below CysForcing: {bool((vs.hi < 0).all())}  "
              f"(max P(baseline better) = {vs.p_better.max():.2f})")
        print(f"  released predictors indistinguishable from one another: "
              f"{bool(((pair.lo <= 0) & (pair.hi >= 0)).all())}")


def main():
    args = parse_args()
    sites = pd.read_csv(args.sites, sep="\t")
    if args.cysforcing_ood or args.cysforcing_ciaa:
        sites = override_cysforcing(sites, args.cysforcing_ood, args.cysforcing_ciaa)
    lab = label(sites)
    os.makedirs(args.out, exist_ok=True)

    radar = radar_table(lab)
    homology = homology_table(lab)
    radar.to_csv(os.path.join(args.out, "fig4_radar.tsv"), sep="\t", index=False, float_format="%.12g")
    homology.to_csv(os.path.join(args.out, "fig4_homology.tsv"), sep="\t", index=False, float_format="%.12g")
    boot = None
    if not args.no_bootstrap:
        boot = bootstrap_table(lab)
        boot.to_csv(os.path.join(args.out, "fig4_bootstrap.tsv"), sep="\t", index=False, float_format="%.12g")
    summary(radar, homology, boot)
    print(f"\nwrote {args.out}/fig4_*.tsv")


if __name__ == "__main__":
    main()
