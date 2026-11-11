"""Precompute the per-protein ESM-2 and SaProt tensors that training reads.

Training and test-set evaluation do not featurize on the fly: they load one
tensor per protein from a cache. This script builds that cache from AlphaFold2
PDB files, writing

    <out>/esm_feat/<UniProt>.pt        [L, 1280]  ESM-2 650M, layer 33
    <out>/saprot_feat/<UniProt>.pt     [L, 1280]  SaProt 650M, last hidden

The encoders are imported from simple_inference rather than reimplemented, so
the features behind a trained model and the features behind a zero-shot
prediction cannot drift apart.

Expect ~24 GB for the 5,314 training proteins and several GPU-hours; the run is
resumable because existing tensors are skipped.
"""
import argparse
import glob
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_inference import ESM_model, SAProt_model, extract_sequence_from_pdb


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-i", "--pdb_folder", required=True,
                   help="directory of <UniProt>.pdb files")
    p.add_argument("-o", "--out", default="data/features",
                   help="output cache root (default: data/features)")
    p.add_argument("--saprot", default="./SaProt_650M_AF2",
                   help="folder holding SaProt_650M_AF2/ and bin/foldseek")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute tensors that already exist")
    return p.parse_args()


def main():
    args = parse_args()
    esm_dir = os.path.join(args.out, "esm_feat")
    sa_dir = os.path.join(args.out, "saprot_feat")
    os.makedirs(esm_dir, exist_ok=True)
    os.makedirs(sa_dir, exist_ok=True)

    pdbs = sorted(glob.glob(os.path.join(args.pdb_folder, "*.pdb")))
    if not pdbs:
        raise SystemExit(f"no *.pdb under {args.pdb_folder}")
    print(f"{len(pdbs)} structures -> {args.out}")

    esm_model = ESM_model()
    saprot_model = SAProt_model(model_folder=args.saprot)

    done = skipped = failed = 0
    for pdb in tqdm(pdbs, desc="featurizing"):
        stem = os.path.splitext(os.path.basename(pdb))[0]
        esm_out = os.path.join(esm_dir, f"{stem}.pt")
        sa_out = os.path.join(sa_dir, f"{stem}.pt")
        if not args.overwrite and os.path.exists(esm_out) and os.path.exists(sa_out):
            skipped += 1
            continue
        try:
            seq = extract_sequence_from_pdb(pdb)
            if not seq:
                raise ValueError("no SEQRES record")
            torch.save(esm_model(seq), esm_out)
            torch.save(saprot_model(pdb), sa_out)
            done += 1
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"  [skip] {stem}: {type(exc).__name__}: {exc}")

    print(f"\nwrote {done}, already present {skipped}, failed {failed}")
    if failed:
        print("Failed entries are usually truncated or chain-less PDB files; "
              "the split CSVs reference every protein, so re-run after fixing "
              "them or training will stop on the missing tensor.")


if __name__ == "__main__":
    main()
