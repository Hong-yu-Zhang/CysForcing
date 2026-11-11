import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def pad_collate(batch):
    """Pad a batch of variable-length proteins to the longest one in the batch.

    Residue-level labels are padded with zeros, which keeps them aligned with
    the features and leaves padded positions unselected: sites are picked by
    `ratio != 0`, and every layer of the model acts position-wise, so padding
    cannot influence a real residue.
    """
    esm_feats, saprot_feats, labels, ratios, cys_masks = zip(*batch)
    max_len = max(f.size(0) for f in esm_feats)

    def pad_1d(t):
        t = t[:max_len]
        return torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=0)

    def pad_2d(t):
        return torch.nn.functional.pad(t, (0, 0, 0, max_len - t.size(0)), value=0)

    return (torch.stack([pad_2d(f) for f in esm_feats]),
            torch.stack([pad_2d(f) for f in saprot_feats]),
            torch.stack([pad_1d(t) for t in labels]),
            torch.stack([pad_1d(t) for t in ratios]),
            torch.stack([pad_1d(t) for t in cys_masks]))
