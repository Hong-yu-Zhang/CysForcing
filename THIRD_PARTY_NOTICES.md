# Third-party notices

CysForcing itself is MIT licensed (see `LICENSE`). The components below are not
ours; each is listed with what this repository contains of it.

## Code derived from or vendored into this repository

**DrugBAN** (MIT, Copyright (c) 2022 Peizhen Bai) —
<https://github.com/peizhenbai/DrugBAN>. The training scaffold is derived from
it; its copyright notice is retained in `LICENSE`.

**SaProt** (MIT) — <https://github.com/westlake-repl/SaProt>.
`foldseek_util.py` is vendored from it (its `utils/foldseek_util.py`), kept
verbatim apart from import paths. Its copyright notice is retained in `LICENSE`.

## Models downloaded at run time, not redistributed

**ESM-2 650M** (MIT) — fetched by `fair-esm` on first use.

**SaProt 650M (AF2)** (MIT) and the **foldseek** binary (GPL-3.0) — downloaded
by the user into `SaProt_650M_AF2/`, see the README. foldseek is invoked as an
external executable and is not linked into this code.

**ESM-C 300M / 600M** (MIT / EvolutionaryScale open licence) — used only for the
backbone sensitivity analysis (`reproduce/run_backbone.sh`); obtain them from
<https://github.com/evolutionaryscale/esm> under their licence terms.

**AlphaFold DB structures** (CC BY 4.0) — fetched by `scripts/get_AF2.py`. The project page and the
images and video in `docs/static/` render the AlphaFold DB model of GSTO1 (UniProt P78417).

## Data distributed via Zenodo

The datasets are not stored in this repository; they are archived on Zenodo and
downloaded into `data/` (see `data/README.md`). The archive is released under
CC BY 4.0, with the exception noted below.

**CIAA AF2-denovo labels** — `data/ciaa/af2_denovo_labels.csv` holds the
`identifier`, `ratio_mean` and `reactivity` columns of
`isotop_alphafold_denovo_set.csv` from the CIAA release (MIT, Copyright (c) 2024
Backus Lab at UCLA), redistributed under that licence (not CC BY 4.0) so that
`reproduce/run_ciaa.sh` runs without a manual download.

**isoTOP-ABPP reactivity ratios** (`data/splits/*.csv`, `data/ood/*.tsv`) are
compiled from published chemoproteomic datasets; see the manuscript Methods for
the source studies.

## Baseline predictors compared against in Fig. 4

Their per-site scores ship as source data in `paper/data/ood_sites.tsv`; none of
their code is redistributed here. How each was applied is described in the
Supplementary Methods.

- **CIAA random forest** — ACS Chemical Biology (2025). Scored on top of a
  reimplementation of its structural feature extractor.
- **HyperCys** — International Journal of Molecular Sciences (2023),
  <https://github.com/mingjie-tech/HyperCys>. The released 34-descriptor
  scikit-learn model with its official feature standardization.
- **sbPCR** — Biochemistry (2017), doi:10.1021/acs.biochem.7b00897. The released
  RBF support-vector model; its signed decision value is used as the score.

The homology-transfer reference uses **DIAMOND**
(<https://github.com/bbuchfink/diamond>, GPL-3.0), invoked as an external
executable.
