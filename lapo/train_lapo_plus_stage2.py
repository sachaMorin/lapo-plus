import torch
import torch.nn.functional as F

import doy
from doy import loop

import config
import paths
import utils
from utils import (
    create_decoder,
    seed_everything,
    get_dataset_splits,
    get_opt_lr_scheduler,
    freeze_module,
)
from eval_utils import entropy_from_logits, test_step_idm, log_confusion_matrix_idm

# config
cfg = config.get()
if cfg.lapo_plus_stage2.idm_only: 
    state_dicts = None
    cfg.stage_exp_name = (
        doy.random_proquint(1)
        + f"-obs{cfg.lapo_plus_stage2.n_observed_samples}"
        + f"-s{cfg.seed}"
    )
    doy.print("[bold green]Running IDM training with config:")
    config.print_cfg(cfg)
    run, logger = config.wandb_init("idm_stage1", config.get_wandb_cfg(cfg))
    desc = "[green bold](idm_stage1) IDM"
    if cfg.lapo_plus_stage2.freeze_backbone:
        raise ValueError("Should not freeze backbone when training IDM from scratch")
else:
    state_dicts = torch.load(
        paths.get_lapo_stage1_path(config.get().exp_name), weights_only=False
    )
    cfg = config.get(base_cfg=state_dicts["cfg"], reload_keys=["lapo_plus_stage2"])
    cfg.stage_exp_name = (
        doy.random_proquint(1)
        + f"-obs{cfg.lapo_plus_stage2.n_observed_samples}"
        + f"-s{cfg.seed}"
        + (f"-freeze-backbone" if cfg.lapo_plus_stage2.freeze_backbone else "")
    )
    doy.print("[bold green]Running LAPO+ stage 2 (IDM Decoding) with config:")
    config.print_cfg(cfg)
    run, logger = config.wandb_init("lapo_plus_stage2", config.get_wandb_cfg(cfg))
    desc = "[green bold](lapo_plus_stage2) Decoding IDM"

seed_everything(cfg.seed)

if not cfg.lapo_plus_stage2.idm_only and state_dicts["step"] != cfg.lapo_stage1.steps:
    doy.log(
        f"[bold red]Warning: using IDM/WM from incomplete training run {state_dicts['step']}/{cfg.lapo_stage1.steps} steps"
    )

# models
idm, _ = utils.create_dynamics_models(cfg.model, state_dicts=state_dicts)
if cfg.lapo_plus_stage2.freeze_backbone:
    freeze_module(idm)

idm.decoder = create_decoder(
    in_dim=cfg.model.la_dim,
    out_dim=cfg.model.ta_dim,
    hidden_sizes=cfg.model.decoder_hidden_sizes,
)

# optimizer
opt, lr_sched = get_opt_lr_scheduler(
    modules=[idm],
    steps=cfg.lapo_plus_stage2.steps,
    lr=cfg.lapo_plus_stage2.lr,
)

# data
train_data, test_data, train_iter, test_iter = get_dataset_splits(
    cfg.env_name, cfg.lapo_plus_stage2.n_observed_samples, cfg.lapo_plus_stage2.bs
)

_, eval_metrics = utils.eval_latent_repr(train_data, idm)
doy.log(f"Decoder metrics sanity check: {eval_metrics}")


# training loop
for step in loop(
    cfg.lapo_plus_stage2.steps + 1, desc=desc
):
    lr_sched.step(step)

    idm.train()
    batch = next(train_iter)
    idm.label(batch)
    pred_ta = batch["pred_ta"]

    ta = batch["ta"][:, -2]
    train_loss_expert = F.cross_entropy(pred_ta, ta)

    opt.zero_grad()
    train_loss_expert.backward()
    opt.step()

    with torch.inference_mode():
        train_acc_expert = (pred_ta.argmax(-1) == ta).float().mean()
        train_entropy = entropy_from_logits(pred_ta).mean().item()

    logger(
        step=step,
        train_loss_expert=train_loss_expert,
        train_acc_expert=train_acc_expert,
        train_entropy=train_entropy,
        **lr_sched.get_state(),
    )

    if step % cfg.test_every == 0:
        test_step_idm(step, idm, test_iter, logger)


# Log confusion matrix
log_confusion_matrix_idm(step, idm, test_iter, logger)


if cfg.lapo_plus_stage2.idm_only:
    out_path = paths.get_idm_stage1_path(
        cfg.exp_name, obs=cfg.lapo_plus_stage2.n_observed_samples, seed=cfg.seed
    )
else:
    out_path=paths.get_lapo_plus_stage2_path(
        cfg.exp_name, obs=cfg.lapo_plus_stage2.n_observed_samples, seed=cfg.seed, freeze_backbone=cfg.lapo_plus_stage2.freeze_backbone
    )

torch.save(
    dict(idm=doy.state_dict_orig(idm), cfg=cfg),
    out_path,
)
