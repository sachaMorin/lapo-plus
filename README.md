# On the Sample Efficiency of Inverse Dynamics Models for Semi-Supervised Imitation Learning: LAPO+ and Procgen Experiments

Code for **LAPO+** and the Procgen experiments from our ICML 2026 paper [On the Sample Efficiency of Inverse Dynamics Models for Semi-Supervised Imitation Learning](https://arxiv.org/abs/2602.02762). 

This codebase was originally a fork of the [LAPO repo](https://github.com/schmidtdominik/LAPO). See also the [LAPO Paper](https://arxiv.org/abs/2312.10812).


## Methods overview

The following methods learn Procgen policies offline using a large unlabeled dataset $\mathcal{D_U}$ and a small action-labeled dataset $\mathcal{D_L}$.

### LAPO

- **Stage 1: LIDM-LFDM** — `train_lapo_stage1.py`. Jointly train a latent inverse dynamics model (LIDM) and a latent forward dynamics model (LFDM) on $\mathcal{D_U}$ using a reconstruction objective.
- **Stage 2: Latent IDM labeling** — `train_lapo_stage2.py`. Train a latent policy on the labels predicted by the LIDM on $\mathcal{D_U}$. 
- **Stage 3: BC with pretrained backbone** — `train_lapo_stage3.py`. Attach a small decoder on top of the latent policy and train on $\mathcal{D_L}$ to map latent actions to true actions.

### LAPO+

LAPO+ keeps Stage 1 unchanged but swaps the order of "decode" and "BC":

- **Stage 1: LIDM-LFDM** — `train_lapo_stage1.py` (shared with LAPO).
- **Stage 2: IDM learning with pretrained backbone** — `train_lapo_plus_stage2.py`. Attach a decoder to the LIDM and train on $\mathcal{D_L}$ to map latent actions to true actions.
- **Stage 3: IDM labeling** — `train_lapo_plus_stage3.py`. Use the action-decoded IDM to generate labels for $\mathcal{D_U}$ and train a policy on those labels.

### BC

- **Stage 1: BC** — `train_lapo_stage3.py` with `lapo_stage3.bc_only=true`. Train a policy from scratch via BC on $\mathcal{D_L}$.

### IDM Labeling

- **Stage 1: IDM learning** — `train_lapo_plus_stage2.py` with `lapo_plus_stage2.idm_only=true`. Train an IDM from scratch on $\mathcal{D_L}$.
- **Stage 2: IDM labeling** — `train_lapo_plus_stage3.py` with `lapo_plus_stage2.idm_only=true`. Use the IDM to generate labels for $\mathcal{D_U}$ and train a policy on those labels.

## Setup

We use Python 3.10. To install dependencies:

```bash
pip install -r requirements.txt

# for unzipping the dataset
sudo apt install unzip
```

> If you use Python 3.11+, replace `procgen` in `requirements.txt` with `procgen-mirror`.

### Dataset

Create a directory or symlink at `lapo/expert_data`, then download the expert data for one or more of the 16 Procgen tasks. Uncomment the tasks you want in `setup_data.sh` and run:

```bash
bash setup_data.sh
```

If Google Drive bandwidth limits block the download, you can fetch the zip files manually from [the dataset folder](https://drive.google.com/drive/folders/1XjpcfOm0NafPYFPnNtoHfhJ4nHVkQSB1) and unzip them under `lapo/expert_data/`. The directory should look like:

```
lapo/expert_data
├── bigfish
│  ├── train
│  └── test
├── bossfight
│  ├── train
│  └── test
...
```

The data is provided as `.npz` chunks containing observations, actions, log-probs, value estimates, etc. By default the loader uses up to 80 chunks (~2.5M frames, ~40 GB host RAM); change `MAX_DATA_CHUNKS` in `lapo/paths.py` to reduce memory usage.

## Running an example environment (bigfish)

All commands below assume you are in the `lapo/` directory:

```bash
cd lapo
```

We use a single `exp_name` to share the LIDM-LFDM checkpoint across LAPO and LAPO+. Hyperparameters live in `lapo/config.yaml` and can be overridden from the command line via [OmegaConf](https://omegaconf.readthedocs.io/) dotted syntax (e.g. `lapo_stage3.lr=2e-4`).

### LAPO

**Stage 1 — LIDM-LFDM:**

```bash
python train_lapo_stage1.py \
    env_name=bigfish \
    seed=11 \
    exp_name=demo \
    lapo_stage1.steps=50000
```

This writes `exp_results/demo/lapo_stage1_idm_fdm.pt`.

**Stage 2 — Latent IDM labeling:**

```bash
python train_lapo_stage2.py \
    env_name=bigfish \
    seed=11 \
    exp_name=demo \
    lapo_stage2.steps=60000
```

This loads the Stage 1 checkpoint and writes `exp_results/demo/lapo_stage2_latent_policy.pt`.

**Stage 3 — BC with pretrained backbone:**

```bash
python train_lapo_stage3.py \
    env_name=bigfish \
    seed=11 \
    exp_name=demo \
    lapo_stage3.n_observed_samples=4000 \
    lapo_stage3.freeze_backbone=true \
    lapo_stage3.lr=2e-4 \
    lapo_stage3.bc_only=false \
    lapo_stage3.steps=10000
```

Key arguments:
- `n_observed_samples` — size of $\mathcal{D_L}$ (number of action-labeled transitions). Use `-1` for the full dataset.
- `freeze_backbone` — if `true`, only the action decoder is trained; if `false`, the latent policy is fine-tuned end-to-end.

### LAPO+

LAPO+ reuses the LAPO Stage 1 checkpoint trained above (same `exp_name`).

**Stage 2 — IDM learning with pretrained backbone:**

```bash
python train_lapo_plus_stage2.py \
    env_name=bigfish \
    seed=11 \
    exp_name=demo \
    lapo_plus_stage2.n_observed_samples=4000 \
    lapo_plus_stage2.freeze_backbone=true \
    lapo_plus_stage2.lr=2e-4 \
    lapo_plus_stage2.idm_only=false \
    lapo_plus_stage2.steps=10000
```

**Stage 3 — IDM labeling:**

```bash
python train_lapo_plus_stage3.py \
    env_name=bigfish \
    seed=11 \
    exp_name=demo \
    lapo_plus_stage2.n_observed_samples=4000 \
    lapo_plus_stage2.freeze_backbone=true \
    lapo_plus_stage2.idm_only=false \
    lapo_plus_stage3.steps=60000
```

Stage 3 reads the matching Stage 2 checkpoint, so the `n_observed_samples`, `freeze_backbone`, and `idm_only` flags must match the values used in Stage 2. Stage-3-specific hyperparameters (e.g. `lapo_plus_stage3.lr`) are read from `config.yaml`.

### BC

**Stage 1 — BC:**

```bash
python train_lapo_stage3.py \
    env_name=bigfish \
    seed=11 \
    exp_name=demo \
    lapo_stage3.n_observed_samples=4000 \
    lapo_stage3.freeze_backbone=false \
    lapo_stage3.lr=2e-4 \
    lapo_stage3.bc_only=true \
    lapo_stage3.steps=120000
```

### IDM Labeling

**Stage 1 — IDM learning:**

```bash
python train_lapo_plus_stage2.py \
    env_name=bigfish \
    seed=11 \
    exp_name=demo \
    lapo_plus_stage2.n_observed_samples=4000 \
    lapo_plus_stage2.freeze_backbone=false \
    lapo_plus_stage2.lr=2e-4 \
    lapo_plus_stage2.idm_only=true \
    lapo_plus_stage2.steps=60000
```

**Stage 2 — IDM labeling:**

```bash
python train_lapo_plus_stage3.py \
    env_name=bigfish \
    seed=11 \
    exp_name=demo \
    lapo_plus_stage2.n_observed_samples=4000 \
    lapo_plus_stage2.freeze_backbone=false \
    lapo_plus_stage2.idm_only=true \
    lapo_plus_stage3.steps=60000
```

The `n_observed_samples`, `freeze_backbone`, and `idm_only` flags must match the values used in Stage 1.


## Logging

All scripts log to [Weights & Biases](https://wandb.ai/) under different projects for each stage  (e.g., `lapo_stage1`, `lapo_stage2`). Set `WANDB_MODE=disabled` to avoid logging.

## SLURM Launch Scripts

Our SLURM launch scripts to train on all environments are available on the `slurm` branch for reference. They will have to be adapted to your environment.

## Citation

If you use this code, please cite both our paper and the original LAPO paper:

```bibtex
@article{morin2026sample,
  title={On the Sample Efficiency of Inverse Dynamics Models for Semi-Supervised Imitation Learning},
  author={Morin, Sacha and Byeon, Moonsub and Jolicoeur-Martineau, Alexia and Lachapelle, S{\'e}bastien},
  journal={arXiv preprint arXiv:2602.02762},
  year={2026}
}
```

```bibtex
@inproceedings{lapo,
  title={Learning to Act without Actions},
  author={Schmidt, Dominik and Jiang, Minqi},
  booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
