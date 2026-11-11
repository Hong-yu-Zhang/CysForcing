# Data

The datasets are archived separately on Zenodo and are **not** stored in this
repository. The code expects them under `data/`:

```
data/
├── splits/{train,val,test}.csv       homology-aware split, one row per protein
├── ood/pred_exp_inter.tsv            Mouse + SL1344 measured ratios
└── ciaa/af2_denovo_val.uniprots.txt  CIAA AF2-denovo accessions
    ciaa/af2_denovo_labels.csv        CIAA AF2-denovo reactivity labels
```

## Download

Get the archive from Zenodo (DOI: 10.5281/zenodo.XXXXXXX). It expands to a
single `CysForcing-data-v1.0/` folder; copy its `data/` payload in next to
`main.py`:

```bash
unzip CysForcing-data-v1.0.zip
cp -r CysForcing-data-v1.0/data/* data/
```

Each subfolder ships its own `README.md` documenting the columns and labelling;
`SHA256SUMS` in the archive lets you verify integrity
(`cd CysForcing-data-v1.0 && sha256sum -c SHA256SUMS`).

## Not included

Protein **structures** are AlphaFold2 models fetched on demand from the
AlphaFold Protein Structure Database by `scripts/get_AF2.py` and the
`reproduce/*.sh` scripts, keyed by the accessions above; they are not part of
the Zenodo archive. The **feature cache** (`data/features/`) is built locally by
`scripts/build_features.py`. Both are git-ignored.

## Figures

Reproducing the paper figures does **not** require this download:
`paper/` ships its own derived source tables in `paper/data/`, so
`bash paper/run_all.sh` works from a clean clone.
