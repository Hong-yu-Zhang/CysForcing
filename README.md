<p align="center">
  <img src="docs/static/banner.png" width="100%" alt="CysForcing: ratio-prior multimodal alignment for generalizable cysteine reactivity prediction">
</p>

<h3 align="center">Ratio-prior Multimodal Alignment for Generalizable Cysteine Reactivity Prediction</h3>

<p align="center">
  Hongyu Zhang, Feng Zhang, Fengzhang Wang, Zihua Liu, Yudi Guo, Zhiwei Nie*, Chu Wang*, Jie Chen*, Yuan Liu*<br>
  <sub>Peking University · Fudan University · Harbin Institute of Technology, Shenzhen · Pengcheng Laboratory &nbsp;·&nbsp; *corresponding authors</sub>
</p>

<p align="center">
  <a href="https://hong-yu-zhang.github.io/CysForcing/"><b>Project page</b></a> &nbsp;·&nbsp;
  <b>Paper</b> <sub>(soon)</sub> &nbsp;·&nbsp;
  <a href="https://doi.org/10.5281/zenodo.24350021"><b>Data</b></a> &nbsp;·&nbsp;
  <a href="#demo"><b>Video</b></a> &nbsp;·&nbsp;
  <a href="https://hong-yu-zhang.github.io/CysForcing/predict/"><b>Web interface</b></a>
</p>

<p align="center">
  <a href="https://hong-yu-zhang.github.io/CysForcing/"><img alt="project page" src="https://img.shields.io/badge/project-page-26c6da?style=flat-square"></a>
  <a href="https://doi.org/10.5281/zenodo.24350021"><img alt="data" src="https://img.shields.io/badge/data-Zenodo-0e7490?style=flat-square"></a>
  <a href="#demo"><img alt="demo" src="https://img.shields.io/badge/demo-2%20min%20video-0891b2?style=flat-square"></a>
  <a href="https://hong-yu-zhang.github.io/CysForcing/predict/"><img alt="web interface" src="https://img.shields.io/badge/web-interface-0891b2?style=flat-square"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10-155e75?style=flat-square">
  <img alt="pytorch" src="https://img.shields.io/badge/pytorch-1.12-155e75?style=flat-square">
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-334155?style=flat-square"></a>
</p>

<table align="center">
  <tr>
    <td align="center" width="25%"><h2>8,738</h2><sub>measured cysteines</sub></td>
    <td align="center" width="25%"><h2>5,302</h2><sub>proteins</sub></td>
    <td align="center" width="25%"><h2>~100</h2><sub>in-house isoTOP-ABPP experiments</sub></td>
    <td align="center" width="25%"><h2>1.9M</h2><sub>trainable parameters</sub></td>
  </tr>
</table>

Code, trained weights and figure scripts for the paper above.

## Overview

CysForcing predicts which cysteines of a protein are reactive from its sequence and AlphaFold2 structure. Two small adapters map frozen ESM-2 and SaProt embeddings into one space and align them residue by residue; measured isoTOP-ABPP competition ratios then act as a prior on that space, so that sites of similar reactivity sit close together. Only 1.9M parameters are trained.

<p align="center"><img src="docs/static/method.png" width="100%" alt="CysForcing architecture"></p>

## Results

| Evaluation | Metric | Strongest baseline | CysForcing | Gain |
|:--|:--:|:--|:--:|:--:|
| Held-out proteins, ≤40% identity to training | F1 | 0.439 (ESM2-3B) | **0.542** | +0.103 |
| Zero-shot, *M. musculus* (606 sites) | AUROC | 0.596 (sbPCR) | **0.872** | +0.276 |
| Zero-shot, *S. typhimurium* SL1344 (317 sites) | AUROC | 0.593 (CIAA RF) | **0.761** | +0.168 |
| Zero-shot, CIAA AF2-denovo (256 sites) | AUROC | 0.555 (CIAA RF) | **0.674** | +0.119 |
| Zero-shot sites with no measured homolog (69–98%) | AUROC | ≈0.50 (homology transfer) | **0.67–0.83** | +0.17 to +0.33 |

Baselines are five protein foundation models (ESM-2 650M/3B/15B, SaProt 650M/1B) on the held-out split, and the released predictors CIAA, HyperCys and sbPCR on the zero-shot sets, scored on the same sites and structures with sites labelled hyper-reactive at ratio ≤ 2.3 and low-reactive at ≥ 10. The paper reports every similarity cutoff, the ablations and the significance tests; `bash paper/run_all.sh` regenerates all of them.

## Demo

<p align="center">
  <a href="https://hong-yu-zhang.github.io/CysForcing/#demo"><img src="docs/static/demo_poster.jpg" width="70%" alt="Two-minute video walkthrough of the results"></a><br>
  <sub>Two-minute walkthrough of the results (plays on the <a href="https://hong-yu-zhang.github.io/CysForcing/">project page</a>)</sub>
</p>

## Web interface

`webapp/` serves CysForcing in the browser from one GPU. Give a UniProt accession, for which the server fetches the AlphaFold DB model, or upload a PDB file: every cysteine is scored, shown along the sequence and on the structure, and can be downloaded as the same `result.json` that `simple_inference.py` writes. It needs nothing beyond the requirements below.

The page itself is [`docs/predict/`](docs/predict/), and the project site hosts it at [hong-yu-zhang.github.io/CysForcing/predict](https://hong-yu-zhang.github.io/CysForcing/predict/). Until a public server is attached there, it shows the four examples, precomputed with the released model.

```bash
python webapp/server.py --gpu 0 --port 7860    # then open http://localhost:7860
```

The models load once; each protein then takes a fraction of a second on the GPU, plus the AlphaFold DB download. `POST /api/predict` with `{"uniprot": "P78417"}` or `{"pdb": "<file text>"}` returns the scores as JSON, and `--host 0.0.0.0` serves other machines. To attach a server to the hosted page, serve it over HTTPS, start it with `--allow-origin https://hong-yu-zhang.github.io`, and put its URL in the `cysforcing-api` meta tag of `docs/predict/index.html`.

<p align="center"><img src="docs/static/webapp.webp" width="100%" alt="The web interface: per-cysteine probabilities of GSTO1 along the sequence, on the AlphaFold2 structure and in a table"></p>

## Getting started

### 1 · Environment

Python ≥ 3.10. Install PyTorch from the CUDA index first, then the requirements:

```bash
conda create -n cysforcing python=3.10 -y && conda activate cysforcing
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

SaProt and foldseek are not redistributed: put [SaProt_650M_AF2](https://huggingface.co/westlake-repl/SaProt_650M_AF2) at `SaProt_650M_AF2/SaProt_650M_AF2/` and the foldseek binary at `SaProt_650M_AF2/bin/foldseek`, under the repository root. ESM-2 downloads on first use.

### 2 · Predict

Every cysteine in a folder of PDB files; the output JSON gives each site a reactivity probability:

```bash
python simple_inference.py --model configs/model/CysForcing.yaml --weight weights/CysForcing.pth \
    --pdb_folder /path/to/pdbs --out_dir result_inference/mine
```

### 3 · Data

The training and evaluation sets are on [Zenodo](https://doi.org/10.5281/zenodo.24350021) ([format](data/README.md)); structures come from AlphaFold DB:

```bash
unzip CysForcing-data-v1.0.zip && cp -r CysForcing-data-v1.0/data/* data/
awk -F',' 'FNR>1{print $1}' data/splits/*.csv | sort -u > /tmp/uniprots.txt
python scripts/get_AF2.py -i /tmp/uniprots.txt -o data/raw/AF2_pdb -j 12
python scripts/build_features.py -i data/raw/AF2_pdb -o data/features    # ~24 GB, resumable
```

### 4 · Train

```bash
python main.py --model configs/model/CysForcing.yaml --data configs/data/cys.yaml
```

### 5 · Reproduce the paper

```bash
python reproduce/eval_homology.py   # held-out proteins at 100 / 80 / 60 / 40 % identity
bash reproduce/run_ood.sh           # zero-shot mouse and SL1344
bash reproduce/run_ciaa.sh          # zero-shot CIAA AF2-denovo
bash paper/run_all.sh               # all figures from paper/data, CPU only
```

[paper/README.md](paper/README.md) describes the figure source data.

## Citation

```bibtex
@article{zhang2026cysforcing,
  title   = {Ratio-prior Multimodal Alignment for Generalizable Cysteine Reactivity Prediction},
  author  = {Zhang, Hongyu and Zhang, Feng and Wang, Fengzhang and Liu, Zihua and Guo, Yudi and
             Nie, Zhiwei and Wang, Chu and Chen, Jie and Liu, Yuan},
  journal = {Under review},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). CysForcing builds on [ESM-2](https://github.com/facebookresearch/esm), [SaProt](https://github.com/westlake-repl/SaProt), [DrugBAN](https://github.com/peizhenbai/DrugBAN), [foldseek](https://github.com/steineggerlab/foldseek) and [AlphaFold DB](https://alphafold.ebi.ac.uk/); [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists their licences and the sources of the released data.