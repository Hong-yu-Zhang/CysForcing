"""Train CysForcing on the cysteine reactivity splits.

    python main.py --model configs/model/CysForcing.yaml --data configs/data/cys.yaml
"""
import argparse
import os
import re
import warnings
from time import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from dataloader import CysDataset
from models import CysForcing
from run.trainer import Trainer
from utils import pad_collate, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train CysForcing")
    p.add_argument('--model', default='configs/model/CysForcing.yaml',
                   help="path to model config file")
    p.add_argument('--data', default='configs/data/cys.yaml',
                   help="path to data config file")
    return p.parse_args()


def create_experiment_directory(result_dir, name):
    """Return a fresh <result_dir>/<name>/expN directory."""
    base = os.path.join(result_dir, name)
    os.makedirs(base, exist_ok=True)
    used = [int(m.group(1)) for d in os.listdir(base)
            if (m := re.fullmatch(r'exp(\d+)', d))
            and os.path.isdir(os.path.join(base, d))]
    path = os.path.join(base, f"exp{max(used) + 1 if used else 0}")
    os.makedirs(path, exist_ok=True)
    return path


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.model)
    cfg.merge_from_file(args.data)
    set_seed(cfg.SOLVER.SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(f"model config: {args.model}")
    print(f"data config : {args.data}")
    print(f"device      : {device}\n")

    loaders = {}
    for split, shuffle, drop_last in (("train", True, True),
                                      ("val", False, False),
                                      ("test", False, False)):
        df = pd.read_csv(os.path.join(cfg.DATA.SPLIT_DIR, f"{split}.csv"))
        loaders[split] = DataLoader(
            CysDataset(df, cfg.DATA.FEAT_ROOT),
            batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=shuffle,
            num_workers=cfg.SOLVER.NUM_WORKERS, drop_last=drop_last,
            collate_fn=pad_collate)

    model = CysForcing(**cfg).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    output_dir = create_experiment_directory(cfg.OUTPUT.DIR, cfg.OUTPUT.RUN_NAME)
    with open(os.path.join(output_dir, "config.yaml"), "w") as fh:
        fh.write(str(cfg))

    trainer = Trainer(model, opt, device, loaders["train"], loaders["val"],
                      loaders["test"], output=output_dir, **cfg)
    result = trainer.train()
    print(f"\nresults written to {output_dir}")
    return result


if __name__ == '__main__':
    start = time()
    main()
    print(f"total running time: {round(time() - start, 2)}s")
