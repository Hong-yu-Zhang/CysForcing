"""Per-cysteine predictions from cached features, written as simple_inference.py would.

simple_inference.py embeds every structure with ESM-2 and SaProt on the fly. This
script reads the tensors from a feature cache instead (<feat_root>/esm_feat and
<feat_root>/saprot_feat, as written by scripts/build_features.py or
scripts/build_esmc_features.py), so a model trained on another sequence backbone
can be scored on the same structures with the same downstream scripts.

    python reproduce/infer_cached.py \
        --model configs/model/CysForcing.yaml --weight weights/CysForcing.pth \
        --feat_root data/features_ood --pdb_folder data/ood/af2_pdb \
        --out result_inference/ood_cached/result.json
"""
import argparse
import glob
import json
import os
import sys

import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs import get_cfg_defaults
from models import CysForcing
from simple_inference import Inference_one, device, extract_sequence_from_pdb
from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="model config")
    p.add_argument("--weight", required=True, help="model weight")
    p.add_argument("--feat_root", required=True,
                   help="cache holding esm_feat/<id>.pt and saprot_feat/<id>.pt")
    p.add_argument("--pdb_folder", required=True,
                   help="structures to score; their SEQRES gives the residue sequence")
    p.add_argument("--out", required=True, help="output result.json")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.model)
    set_seed(cfg.SOLVER.SEED)

    model = CysForcing(**cfg)
    model.load_state_dict(torch.load(args.weight, map_location="cpu"))
    model.to(device)
    model.eval()

    result_list, missing = [], []
    for pdb in tqdm(sorted(glob.glob(os.path.join(args.pdb_folder, "*.pdb")))):
        name = os.path.basename(pdb)
        stem = os.path.splitext(name)[0]
        esm_path = os.path.join(args.feat_root, "esm_feat", f"{stem}.pt")
        sa_path = os.path.join(args.feat_root, "saprot_feat", f"{stem}.pt")
        if not (os.path.exists(esm_path) and os.path.exists(sa_path)):
            missing.append(stem)
            continue
        feat = {"ESM": torch.load(esm_path), "SAProt": torch.load(sa_path)}
        with torch.no_grad():
            binary = Inference_one(model, feat).cpu().tolist()
        seq = extract_sequence_from_pdb(pdb)[:1023]
        c_scores = {f"C_{i+1}": score for i, (residue, score) in enumerate(zip(seq, binary))
                    if residue == "C"}
        c_scores["Sequence"] = seq
        result_list.append({name: c_scores})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result_list, fh, indent=4, ensure_ascii=False)
    print(f"scored {len(result_list)} structures, {len(missing)} without cached features")
    if missing:
        print("  e.g. " + ", ".join(missing[:5]))
    print(f"Wrote predictions: {args.out}")


if __name__ == "__main__":
    main()
