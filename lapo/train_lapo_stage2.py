import torch
import torch.nn.functional as F

import doy
from doy import loop

import config
import paths
import utils
from utils import seed_everything, get_opt_lr_scheduler, get_dataset_splits, freeze_module

# config
state_dicts = torch.load(
    paths.get_lapo_stage1_path(config.get().exp_name), weights_only=False
)
cfg = config.get(base_cfg=state_dicts["cfg"], reload_keys=["lapo_stage2"])
cfg.stage_exp_name = doy.random_proquint(1)
doy.print("[bold green]Running LAPO stage 2 (latent behavior cloning) with config:")
config.print_cfg(cfg)
run, logger = config.wandb_init("lapo_stage2", config.get_wandb_cfg(cfg))

seed_everything(cfg.seed)

if state_dicts["step"] != cfg.lapo_stage1.steps:
    doy.log(
        f"[bold red]Warning: using IDM/WM from incomplete training run {state_dicts['step']}/{cfg.lapo_stage1.steps} steps"
    )

# models
idm, _ = utils.create_dynamics_models(cfg.model, state_dicts=state_dicts)
idm.eval()
freeze_module(idm)
policy = utils.create_policy(cfg.model, cfg.model.la_dim)

# optimizer
opt, lr_sched = get_opt_lr_scheduler(
    modules=[policy],
    steps=cfg.lapo_stage2.steps,
    lr=cfg.lapo_stage2.lr,
)

# data
train_data, test_data, train_iter, test_iter = get_dataset_splits(
    cfg.env_name, -1, cfg.lapo_stage2.bs
)

_, eval_metrics = utils.eval_latent_repr(train_data, idm)
doy.log(f"Decoder metrics sanity check: {eval_metrics}")


# training loop
for step in loop(
    cfg.lapo_stage2.steps + 1,
    desc="[green bold](lapo_stage2) Training latent policy via BC",
):
    lr_sched.step(step)

    policy.train()
    batch = next(train_iter)

    with torch.no_grad():
        idm.label(batch)

    preds = policy(batch["obs"][:, -2])  # the -2 selects last the pre-transition ob
    loss = F.mse_loss(preds, batch["la"])

    opt.zero_grad()
    loss.backward()
    opt.step()

    logger(
        step=step,
        loss=loss,
        **lr_sched.get_state(),
    )

    if step % cfg.test_every == 0:
        policy.eval()
        test_batch = next(test_iter)
        idm.label(test_batch)
        test_loss = F.mse_loss(policy(test_batch["obs"][:, -2]), test_batch["la"])
        logger(step=step, test_loss=test_loss)

torch.save(
    dict(policy=doy.state_dict_orig(policy), cfg=cfg, logger=logger),
    paths.get_lapo_stage2_path(cfg.exp_name),
)
