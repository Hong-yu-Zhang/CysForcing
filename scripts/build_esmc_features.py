"""Precompute per-residue ESM C embeddings as a drop-in for the ESM-2 cache.

Writes <out>/esm_feat/<UniProt>.pt, [L, d_model], from the final hidden state of
an ESM C checkpoint in Hugging Face format (biohub/ESMC-300M, -600M, -6B; needs
transformers >= 5.16, so run it from an environment separate from the one that
loads SaProt and fair-esm).

The encoder sees up to 2046 residues, the most its 2048-token window holds, but
only the first 1023 positions are kept: the model adds these features to the
SaProt ones residue by residue, and SaProt stops at 1023.

    python scripts/build_esmc_features.py -i data/raw/AF2_pdb \
        -o data/features_esmc300m/train --model_dir ESMC/ESMC-300M
"""
import argparse
import glob
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_inference import extract_sequence_from_pdb

CONTEXT_RESIDUES = 2046
KEPT_RESIDUES = 1023


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-i", "--pdb_folder", required=True,
                   help="directory of <UniProt>.pdb files")
    p.add_argument("-o", "--out", required=True, help="output cache root")
    p.add_argument("--model_dir", required=True, help="ESM C checkpoint folder")
    p.add_argument("--shard", type=int, default=0, help="index of this shard")
    p.add_argument("--num_shards", type=int, default=1,
                   help="split the structures across this many processes")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute tensors that already exist")
    return p.parse_args()


class ESMC_model:
    def __init__(self, model_dir, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir, dtype=torch.float32)
        self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, seq):
        inputs = self.tokenizer(seq[:CONTEXT_RESIDUES], return_tensors="pt").to(self.device)
        hidden = self.model(**inputs).last_hidden_state[0]
        # Position 0 is the BOS token.
        return hidden[1:min(len(seq), KEPT_RESIDUES) + 1].float().cpu()


def main():
    args = parse_args()
    esm_dir = os.path.join(args.out, "esm_feat")
    os.makedirs(esm_dir, exist_ok=True)

    pdbs = sorted(glob.glob(os.path.join(args.pdb_folder, "*.pdb")))
    if not pdbs:
        raise SystemExit(f"no *.pdb under {args.pdb_folder}")
    pdbs = pdbs[args.shard::args.num_shards]
    print(f"{len(pdbs)} structures (shard {args.shard}/{args.num_shards}) -> {esm_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = ESMC_model(args.model_dir, device)

    done = skipped = failed = 0
    for pdb in tqdm(pdbs, desc="featurizing"):
        stem = os.path.splitext(os.path.basename(pdb))[0]
        out = os.path.join(esm_dir, f"{stem}.pt")
        if not args.overwrite and os.path.exists(out):
            skipped += 1
            continue
        try:
            seq = extract_sequence_from_pdb(pdb)
            if not seq:
                raise ValueError("no SEQRES record")
            torch.save(encoder(seq), out)
            done += 1
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"  [skip] {stem}: {type(exc).__name__}: {exc}")

    print(f"\nwrote {done}, already present {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
