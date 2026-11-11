# Paper figures

Every data-driven figure of the manuscript is regenerated from the source data
in `paper/data/` by one script each. None of them needs a GPU, structures or the
feature cache:

```bash
bash paper/run_all.sh        # writes paper/out/
```

| Figure | Script | Source data |
|---|---|---|
| Fig. 2 | `fig2.py` | `fig2_benchmark.tsv`, `fig2_training.tsv` |
| Fig. 3 | `fig3.py` | `fig3_ablation.tsv`, `fig3_interpret.npz` |
| Fig. 4 | `fig4_metrics.py`, then `fig4.py` | `ood_sites.tsv` |
| Supplementary Fig. S1 | `figS1.py` | `figS1_backbones.tsv` |

Fig. 1 is a schematic and has no source data.

## Fig. 2 and Fig. 3

`fig2_benchmark.tsv` and `fig3_ablation.tsv` hold the metric values reported in
the manuscript for every method at the 100 / 80 / 60 / 40 % similarity cutoffs.
The CysForcing rows are recomputed from the released weights by
`reproduce/eval_homology.py`. `fig2_training.tsv` is the per-epoch training log
of the released model; the selected checkpoint is epoch 3.

`fig3_interpret.npz` holds the inputs of Fig. 3e,f: the PCA projection of the
fused per-Cys embeddings of the held-out sites with their local mean ratio, the
bootstrap distribution of the ratio-ordering score rho, and its permuted-ratio
null (Methods, interpretability analysis).

## Fig. 4

`fig4_metrics.py` labels the sites of `ood_sites.tsv`, computes the metrics and
the significance tests, and prints the numbers quoted in the text; `fig4.py`
draws the figure from its tables.

Labelling follows the hyper-reactivity convention of the three compared
predictors: `ratio <= 2.3` is hyper-reactive, `ratio >= 10` low-reactive. This
gives 606 Mouse, 317 SL1344 and 476 CIAA sites. Probability-valued methods are
thresholded at 0.5; sbPCR uses its own predicted label.

`ood_sites.tsv` has one row per cysteine scored by all four methods:

| Column | Meaning |
|---|---|
| `set` | `Mouse`, `SL1344` or `CIAA` |
| `uniprot`, `pos0` | protein accession and 0-indexed cysteine position |
| `ratio` | measured isoTOP-ABPP competition ratio |
| `transferable` | the site's best DIAMOND hit in the training set aligns it to a measured training cysteine |
| `transferred_ratio` | that training cysteine's measured ratio (empty when not transferable) |
| `cysforcing` | CysForcing P(reactive), released weights |
| `ciaa_rf` | CIAA random forest P(reactive), on a reimplementation of its structural feature extractor |
| `hypercys` | HyperCys P(reactive), released 34-descriptor model with its official feature standardization |
| `sbpcr` | sbPCR signed SVM decision value, released RBF model |
| `sbpcr_pred` | sbPCR native predicted label |

All four methods see the same sites and the same AlphaFold structures. The
`cysforcing` column is reproducible from this repository: run
`reproduce/run_ood.sh` and `reproduce/run_ciaa.sh`, then score the fresh
predictions in place of the shipped ones,

```bash
python paper/fig4_metrics.py \
    --cysforcing-ood  result_inference/ood/pred_exp_inter_with_prob.tsv \
    --cysforcing-ciaa result_inference/ciaa_denovo/ciaa_denovo_per_site.tsv
```

The three released predictors and the homology search are third-party software
and are not redistributed here; their scores are shipped as source data. See the
Supplementary Methods for how each was applied, and the original publications:

- CIAA: *ACS Chemical Biology* (2025)
- HyperCys: *International Journal of Molecular Sciences* (2023),
  <https://github.com/mingjie-tech/HyperCys>
- sbPCR: *Biochemistry* (2017), doi:10.1021/acs.biochem.7b00897

## Supplementary Fig. S1

`figS1_backbones.tsv` is the `reproduce/eval_homology.py` output for the
released ESM-2 650M model and for the ESM-C 300M / 600M backbones trained by
`reproduce/run_backbone.sh esmc300m` and `reproduce/run_backbone.sh esmc600m`.
