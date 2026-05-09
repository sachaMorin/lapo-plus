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
from eval_utils import (
    test_step_policy,
    log_confusion_matrix_policy,
    entropy_from_logits,
)
from rollout_utils import rollout_metrics

# config
cfg = config.get()

if cfg.lapo_stage3.bc_only:
    cfg.stage_exp_name = (
        doy.random_proquint(1)
        + f"-obs{cfg.lapo_stage3.n_observed_samples}"
        + f"-s{cfg.seed}"
    )
    doy.print("[bold green]Running BC with config:")
    config.print_cfg(cfg)
    run, logger = config.wandb_init("bc_stage1", config.get_wandb_cfg(cfg))
    policy_state = None
    desc = "[green bold](bc_stage1) BC"
    if cfg.lapo_stage3.freeze_backbone:
        raise ValueError("Should not freeze backbone when training BC from scratch")
else:
    state_dict = torch.load(
        paths.get_lapo_stage2_path(config.get().exp_name), weights_only=False
    )
    cfg = config.get(base_cfg=state_dict["cfg"], reload_keys=["lapo_stage3"])
    cfg.stage_exp_name += (
        doy.random_proquint(1)
        + f"-obs{cfg.lapo_stage3.n_observed_samples}"
        + f"-s{cfg.seed}"
        + (f"-freeze-backbone" if cfg.lapo_stage3.freeze_backbone else "")
    )
    doy.print("[bold green]Running LAPO stage 3_bc (policy decoding with BC) with config:")
    config.print_cfg(cfg)
    run, logger = config.wandb_init("lapo_stage3", config.get_wandb_cfg(cfg))
    policy_state = state_dict["policy"]
    desc = "[green bold](lapo_stage3) Offline Decoding Policy via BC"

seed_everything(cfg.seed)

# models
policy = utils.create_policy(
    cfg.model,
    action_dim=cfg.model.la_dim,
    state_dict=policy_state,
    strict_loading=True,
)
if cfg.lapo_stage3.freeze_backbone:
    freeze_module(policy)

policy.decoder = create_decoder(
    in_dim=cfg.model.la_dim,
    out_dim=cfg.model.ta_dim,
    hidden_sizes=cfg.model.decoder_hidden_sizes,
)


# optimizer
opt, lr_sched = get_opt_lr_scheduler(
    modules=[policy],
    steps=cfg.lapo_stage3.steps,
    lr=cfg.lapo_stage3.lr,
)

# data
train_data, test_data, train_iter, test_iter = get_dataset_splits(
    cfg.env_name, cfg.lapo_stage3.n_observed_samples, cfg.lapo_stage3.bs
)


# training loop
for step in loop(
    cfg.lapo_stage3.steps + 1,
    desc=desc,
):
    lr_sched.step(step)

    policy.train()
    batch = next(train_iter)

    pred_la = policy(batch["obs"][:, -2])  # the -2 selects last the pre-transition ob
    pred_ta = policy.decoder(pred_la)
    ta = batch["ta"][:, -2]
    train_loss_expert = F.cross_entropy(pred_ta, ta)

    opt.zero_grad()
    train_loss_expert.backward()
    opt.step()

    with torch.inference_mode():
        train_acc_expert = (pred_ta.argmax(dim=-1) == ta).float().mean().item()
        train_entropy = entropy_from_logits(pred_ta).mean().item()

    logger(
        step=step,
        train_loss_expert=train_loss_expert,
        train_acc_expert=train_acc_expert,
        train_entropy=train_entropy,
        **lr_sched.get_state(),
    )

    if step % cfg.test_every == 0:
        # policy evaluation on expert data
        test_step_policy(
            step=step,
            policy=policy,
            test_iter=test_iter,
            logger=logger,
        )

    if step % cfg.rollout_policy_every == 0:
        rollout_metrics(
            step=step,
            env_name=cfg.env_name,
            gamma=cfg.lapo_stage3.gamma,
            policy=policy,
            num_envs=cfg.lapo_stage3.num_envs,
            logger=logger,
        )

log_confusion_matrix_policy(
    step=step,
    policy=policy,
    test_iter=test_iter,
    logger=logger,
)

if cfg.lapo_stage3.bc_only:
    out_path = paths.get_bc_stage1_path(
        cfg.exp_name, obs=cfg.lapo_stage3.n_observed_samples, seed=cfg.seed
    )
else:
    out_path = paths.get_lapo_stage3_path(
        cfg.exp_name, obs=cfg.lapo_stage3.n_observed_samples, seed=cfg.seed, freeze_backbone=cfg.lapo_stage3.freeze_backbone
    )
torch.save(
    {
        "policy": policy.state_dict(),
        "cfg": cfg,
    },
    out_path,
)
