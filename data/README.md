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

Get the data archive from Zenodo (DOI: 10.5281/zenodo.XXXXXXX). It expands to a
single `CysForcing-data-v1.0/` folder; copy its `data/` payload in next to
`main.py`:

```bash
unzip CysForcing-data-v1.0.zip
cp -r CysForcing-data-v1.0/data/* data/
```

Each subfolder ships its own `README.md` documenting the columns and labelling;
`SHA256SUMS` in the archive lets you verify integrity
(`cd CysForcing-data-v1.0 && sha256sum -c SHA256SUMS`).

## Structures and features

Protein **structures** (AlphaFold2 models, PDB) are provided as a companion
archive, `CysForcing-structures-v1.0.zip`, in the same Zenodo record (8,444
files keyed by the accessions above). Unpack each subfolder where the code
expects it:

```bash
unzip CysForcing-structures-v1.0.zip
cp -r CysForcing-structures-v1.0/splits/* data/raw/AF2_pdb/
cp -r CysForcing-structures-v1.0/ood/*    data/ood/af2_pdb/
cp -r CysForcing-structures-v1.0/ciaa/*   data/ciaa/af2_denovo/
```

Alternatively `scripts/get_AF2.py` and the `reproduce/*.sh` scripts fetch the
structures on demand from the AlphaFold DB (which is versioned, so a model
released after ours may differ slightly). The **feature cache**
(`data/features/`) is built locally by `scripts/build_features.py`. Both the
unpacked structures and the feature cache are git-ignored.

## Figures

Reproducing the paper figures does **not** require this download:
`paper/` ships its own derived source tables in `paper/data/`, so
`bash paper/run_all.sh` works from a clean clone.
