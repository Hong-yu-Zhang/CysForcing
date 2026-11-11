import ast
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils import data
from tqdm import tqdm


def feature_paths(uniprot_ids, feat_root):
    """Locate the cached tensors for a list of UniProt accessions.

    scripts/build_features.py writes one file per protein per encoder, named
    after the accession.
    """
    feat_root = os.path.expanduser(feat_root)
    esm = [os.path.join(feat_root, "esm_feat", f"{u}.pt") for u in uniprot_ids]
    saprot = [os.path.join(feat_root, "saprot_feat", f"{u}.pt") for u in uniprot_ids]
    missing = [p for p in esm + saprot if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(esm) + len(saprot)} feature tensors are "
            f"missing under {feat_root}, e.g. {missing[0]}. "
            f"Build them with scripts/build_features.py.")
    return esm, saprot


class CysDataset(data.Dataset):
    """Per-protein ESM-2 and SaProt features with per-residue labels.

    Labels are stored one list per protein, aligned to the residue sequence:
    `Label` is the binary reactivity call, `Label_reg` the measured depletion
    ratio (non-zero exactly at measured cysteines, which is what selects the
    evaluated sites), and `Cys` marks cysteine positions.
    """

    def __init__(self, df, feat_root, num_threads=32):
        self.df = df.reset_index(drop=True)
        esm_paths, saprot_paths = feature_paths(self.df["UniProt ID"], feat_root)

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            self.esm_feat = list(tqdm(pool.map(torch.load, esm_paths),
                                      total=len(esm_paths), desc="load ESM"))
            self.saprot_feat = list(tqdm(pool.map(torch.load, saprot_paths),
                                         total=len(saprot_paths), desc="load SaProt"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = torch.tensor(ast.literal_eval(row["Label"]))
        ratio = torch.tensor(ast.literal_eval(row["Label_reg"]))
        cys_mask = torch.tensor(ast.literal_eval(row["Cys"]))
        return self.esm_feat[index], self.saprot_feat[index], label, ratio, cys_mask
