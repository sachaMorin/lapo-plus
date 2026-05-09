import torch
import torch.nn.functional as F

from doy import loop

import config
import doy
import paths
import utils
from utils import (
    create_decoder,
    seed_everything,
    get_dataset_splits,
    get_opt_lr_scheduler,
    freeze_module,
)
from rollout_utils import rollout_metrics
from eval_utils import (
    test_step_policy,
    log_confusion_matrix_policy,
    entropy_from_logits,
)

# config
initial_cfg = config.get()
if initial_cfg.lapo_plus_stage2.idm_only:
    state_dict = torch.load(
        paths.get_idm_stage1_path(
            initial_cfg.exp_name,
            obs=initial_cfg.lapo_plus_stage2.n_observed_samples,
            seed=initial_cfg.seed,
        ),
        weights_only=False,
    )
    cfg = config.get(base_cfg=state_dict["cfg"], reload_keys=["lapo_plus_stage3"])
    cfg.stage_exp_name += "-" + doy.random_proquint(1) + f"-s{cfg.seed}" 
    doy.print(
        "[bold green]Running IDM Labeling stage 2 (policy training with IDM labels) with config:"
    )
    config.print_cfg(cfg)
    run, logger = config.wandb_init("idm_stage2", config.get_wandb_cfg(cfg))
    desc = "[green bold](idm_stage2) Training Policy via IDM Relabeling"
else:
    state_dict = torch.load(
        paths.get_lapo_plus_stage2_path(
            initial_cfg.exp_name,
            obs=initial_cfg.lapo_plus_stage2.n_observed_samples,
            seed=initial_cfg.seed,
            freeze_backbone=initial_cfg.lapo_plus_stage2.freeze_backbone,
        ),
        weights_only=False,
    )
    cfg = config.get(base_cfg=state_dict["cfg"], reload_keys=["lapo_plus_stage3"])
    cfg.stage_exp_name += doy.random_proquint(1) + f"-s{cfg.seed}" 
    doy.print(
        "[bold green]Running LAPO stage 3 (policy training with IDM labels) with config:"
    )
    config.print_cfg(cfg)
    run, logger = config.wandb_init("lapo_plus_stage3", config.get_wandb_cfg(cfg))
    desc = "[green bold](lapo_plus_stage3) Training Policy via IDM Relabeling"

seed_everything(cfg.seed)

# models
policy = utils.create_policy(
    cfg.model,
    action_dim=cfg.model.la_dim,
    state_dict=None,
)
policy.decoder = create_decoder(
    in_dim=cfg.model.la_dim,
    out_dim=cfg.model.ta_dim,
    hidden_sizes=cfg.model.decoder_hidden_sizes,
)

idm, _ = utils.create_dynamics_models(cfg.model, state_dicts=None)
idm.decoder = create_decoder(
    in_dim=cfg.model.la_dim,
    out_dim=cfg.model.ta_dim,
    hidden_sizes=cfg.model.decoder_hidden_sizes,
)
idm.load_state_dict(state_dict["idm"])
freeze_module(idm)
idm.eval()


# optimizer
opt, lr_sched = get_opt_lr_scheduler(
    modules=[policy],
    steps=cfg.lapo_plus_stage3.steps,
    lr=cfg.lapo_plus_stage3.lr,
)

# data
train_data, test_data, train_iter, test_iter = get_dataset_splits(
    cfg.env_name, -1, cfg.lapo_plus_stage3.bs
)

# training loop
for step in loop(
    cfg.lapo_plus_stage3.steps + 1,
    desc="[green bold](lapo_plus_stage3) Training Policy via IDM Relabeling",
):
    lr_sched.step(step)

    policy.train()
    batch = next(train_iter)

    with torch.no_grad():
        idm.label(batch)

    pred_la = policy(batch["obs"][:, -2])  # the -2 selects last the pre-transition ob
    pred_ta = policy.decoder(pred_la)

    idm_pred = batch["pred_ta"]
    train_loss_idm = F.cross_entropy(pred_ta, idm_pred.argmax(-1))

    opt.zero_grad()
    train_loss_idm.backward()
    opt.step()

    # more metrics
    with torch.inference_mode():
        train_acc_idm = (
            (pred_ta.argmax(dim=-1) == idm_pred.argmax(-1)).float().mean().item()
        )
        ta = batch["ta"][:, -2]
        train_loss_expert = F.cross_entropy(pred_ta, ta)
        train_acc_expert = (pred_ta.argmax(-1) == ta).float().mean()
        train_entropy = entropy_from_logits(pred_ta).mean().item()

    logger(
        step=step,
        train_loss_idm=train_loss_idm,
        train_acc_idm=train_acc_idm,
        train_loss_expert=train_loss_expert,
        train_acc_expert=train_acc_expert,
        train_entropy=train_entropy,
        **lr_sched.get_state(),
    )

    if step % cfg.test_every == 0:
        test_step_policy(
            step=step,
            policy=policy,
            test_iter=test_iter,
            logger=logger,
            idm=idm,
        )

    if step % cfg.rollout_policy_every == 0:
        rollout_metrics(
            step=step,
            env_name=cfg.env_name,
            gamma=cfg.lapo_plus_stage3.gamma,
            policy=policy,
            num_envs=cfg.lapo_plus_stage3.num_envs,
            logger=logger,
        )

log_confusion_matrix_policy(
    step=step,
    policy=policy,
    test_iter=test_iter,
    logger=logger,
    idm=idm,
)

if cfg.lapo_plus_stage2.idm_only:
    out_path = paths.get_idm_stage2_path(
        cfg.exp_name, obs=initial_cfg.lapo_plus_stage2.n_observed_samples, seed=cfg.seed
    )
else:
    out_path = paths.get_lapo_plus_stage3_path(
        cfg.exp_name, obs=initial_cfg.lapo_plus_stage2.n_observed_samples, seed=cfg.seed, freeze_backbone=cfg.lapo_plus_stage2.freeze_backbone
    )

torch.save(
    {
        "policy": policy.state_dict(),
        "cfg": cfg,
    },
    out_path,
)
