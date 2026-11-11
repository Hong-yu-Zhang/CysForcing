import torch
import torch.nn as nn
import torch.nn.functional as F


def _dropout(p):
    """A dropout layer only when it does something, so dropout 0 keeps the
    module indices, and therefore the state-dict keys, of released weights."""
    return [nn.Dropout(p)] if p > 0 else []


class CysForcing(nn.Module):
    """Dual-encoder cysteine reactivity predictor.

    Per-residue ESM-2 and SaProt features are projected into a shared space by
    one MLP each, summed, and classified. Each branch is also classified on its
    own, which is what lets the ratio prior act on a modality before the two
    are fused.
    """

    def __init__(self, **config):
        super(CysForcing, self).__init__()

        esm_in_dim = config["ESM"]["IN_DIM"]
        esm_hidden_dim = config["ESM"]["HIDDEN_DIM"]
        esm_target_dim = config["ESM"]["TARGET_DIM"]

        saprot_in_dim = config["SAPROT"]["IN_DIM"]
        saprot_hidden_dim = config["SAPROT"]["HIDDEN_DIM"]
        saprot_target_dim = config["SAPROT"]["TARGET_DIM"]

        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        num_classes = config["DECODER"]["NUM_CLASSES"]
        dropout = config["DECODER"].get("DROPOUT", 0.0)

        self.in_mlp_esm = nn.Sequential(
            nn.LayerNorm(esm_in_dim),
            nn.Linear(esm_in_dim, esm_hidden_dim),
            nn.ReLU(),
            *_dropout(dropout),
            nn.Linear(esm_hidden_dim, esm_target_dim)
        )

        self.in_mlp_saprot = nn.Sequential(
            nn.LayerNorm(saprot_in_dim),
            nn.Linear(saprot_in_dim, saprot_hidden_dim),
            nn.ReLU(),
            *_dropout(dropout),
            nn.Linear(saprot_hidden_dim, saprot_target_dim)
        )

        self.aux_head_esm = nn.Sequential(
            nn.ReLU(),
            nn.Linear(esm_target_dim, num_classes)
        )
        self.aux_head_saprot = nn.Sequential(
            nn.ReLU(),
            nn.Linear(saprot_target_dim, num_classes)
        )

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim,
                                         num_classes=num_classes, dropout=dropout)

    def forward(self, esm_feat, saprot_feat, mask=None):
        """`mask` is accepted for interface compatibility and is not used: every
        layer here acts position-wise, so padded positions cannot influence
        real ones."""
        v_esm = self.in_mlp_esm(esm_feat)
        v_saprot = self.in_mlp_saprot(saprot_feat)
        v_fused = v_esm + v_saprot

        # The fused classifier is last; the two per-modality heads precede it.
        binary_list = [self.aux_head_esm(v_esm),
                       self.aux_head_saprot(v_saprot),
                       self.mlp_classifier(v_fused)]

        return {
            'binary': binary_list,
            'feat': [v_esm, v_saprot, v_fused],
        }


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_classes=1, dropout=0.0):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            *_dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            *_dropout(dropout),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, x):
        return self.fc1(x)


def binary_cross_entropy(pred_output, labels, pos_weight=None, label_smooth=0.0):
    m = nn.Sigmoid()
    n = m(pred_output)
    if pos_weight is None and not label_smooth:
        return n, nn.BCELoss()(n, labels)
    if label_smooth:
        labels = labels * (1 - label_smooth) + 0.5 * label_smooth
    weight = None if pos_weight is None else torch.as_tensor(
        pos_weight, dtype=pred_output.dtype, device=pred_output.device)
    loss = F.binary_cross_entropy_with_logits(pred_output, labels, pos_weight=weight)
    return n, loss


def ratio_similarity(ratio, scale=1.0):
    """exp(-|r_i - r_j| / scale): pairs with similar depletion ratios score near 1.

    Ratios run up to 15, so at scale 1 only near-identical pairs carry weight;
    a larger scale widens the band the prior treats as similar.
    """
    return torch.exp(-torch.abs(ratio.unsqueeze(1) - ratio.unsqueeze(0)) / scale)


def ratio_prior_loss(seq_feat, struct_feat, ratio, scale=1.0):
    """Pull the two modalities together in proportion to ratio similarity.

    Every cross-modal pair is penalised by its squared distance weighted by how
    close the two sites' measured ratios are, so the fused space is forced to
    order sites by reactivity rather than only separate reactive from
    unreactive.
    """
    weight = ratio_similarity(ratio, scale)
    distance = torch.cdist(seq_feat, struct_feat, p=2)
    return (weight * distance ** 2).mean()


def cysforcing_loss(z_seq, z_struct, ratio, temp=0.2, lambda_ratio=0.5, ratio_scale=1.0):
    """NT-Xent alignment of the two modalities plus the ratio prior.

    Returns the combined loss and its two components, so training can log the
    contrastive and ratio terms separately.
    """
    batch_size = z_seq.size(0)

    z_i = F.normalize(z_seq, p=2, dim=1)
    z_j = F.normalize(z_struct, p=2, dim=1)
    z = torch.cat([z_i, z_j], dim=0)
    similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    # The two off-diagonals hold the matched sequence/structure pairs.
    positives = torch.cat([torch.diag(similarity, batch_size),
                           torch.diag(similarity, -batch_size)], dim=0)
    off_diagonal = (~torch.eye(batch_size * 2, batch_size * 2,
                               dtype=torch.bool, device=z_seq.device)).float()

    numerator = torch.exp(positives / temp)
    denominator = off_diagonal * torch.exp(similarity / temp)
    contrastive = torch.sum(-torch.log(numerator / torch.sum(denominator, dim=1)))
    contrastive = contrastive / (2 * batch_size)

    ratio_term = ratio_prior_loss(z_seq, z_struct, ratio, ratio_scale)
    return contrastive + lambda_ratio * ratio_term, contrastive, ratio_term
