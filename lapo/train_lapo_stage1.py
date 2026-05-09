import torch

import doy
from doy import loop

import paths
import utils
import config
from utils import seed_everything, get_opt_lr_scheduler, get_dataset_splits

# config
cfg = config.get()
doy.print("[bold green]Running LAPO stage 1 (IDM/FDM training) with config:")
config.print_cfg(cfg)
run, logger = config.wandb_init("lapo_stage1", config.get_wandb_cfg(cfg))

seed_everything(cfg.seed)

# models
idm, wm = utils.create_dynamics_models(cfg.model)

# optimizer
opt, lr_sched = get_opt_lr_scheduler(
    modules=[idm, wm],
    lr=cfg.lapo_stage1.lr,
    steps=cfg.lapo_stage1.steps,
    warmup_steps=50,
    init_lr_scale=0.1,
    final_lr_scale=0.01,
)

# data
train_data, test_data, train_iter, test_iter = get_dataset_splits(
    cfg.env_name, -1, cfg.lapo_stage1.bs
)


# training loop
def train_step():
    idm.train()
    wm.train()

    lr_sched.step(step)

    batch = next(train_iter)

    vq_loss, vq_perp = idm.label(batch)
    wm_loss = wm.label(batch)
    loss = wm_loss + vq_loss

    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([*idm.parameters(), *wm.parameters()], 2)
    opt.step()

    logger(
        step,
        wm_loss=wm_loss,
        global_step=step * cfg.lapo_stage1.bs,
        vq_perp=vq_perp,
        vq_loss=vq_loss,
        grad_norm=grad_norm,
        **lr_sched.get_state(),
    )


def test_step():
    idm.eval()  # disables idm.vq ema update
    wm.eval()

    # evaluate IDM + FDM generalization on (action-free) test data
    batch = next(test_iter)
    idm.label(batch)
    wm_loss = wm.label(batch)

    # train latent -> true action decoder and evaluate its predictiveness
    _, eval_metrics = utils.eval_latent_repr(train_data, idm)

    logger(
        step,
        wm_loss_test=wm_loss,
        global_step=step * cfg.lapo_stage1.bs,
        **eval_metrics,
    )


for step in loop(
    cfg.lapo_stage1.steps + 1, desc="[green bold](lapo_stage1) Training IDM + FDM"
):
    train_step()

    # fix test frequency since more expensive than other scripts
    if step % 500 == 0:
        test_step()

    if step > 0 and (step % 5_000 == 0 or step == cfg.lapo_stage1.steps):
        torch.save(
            dict(
                **doy.get_state_dicts(wm=wm, idm=idm, opt=opt),
                step=step,
                cfg=cfg,
                logger=logger,
            ),
            paths.get_lapo_stage1_path(cfg.exp_name),
        )
