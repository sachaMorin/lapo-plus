import config

import numpy as np
import torch
from torch.distributions import Categorical

import env_utils
from data_loader import normalize_obs


def rollout(policy, envs):
    """Rollout policy and compute episodic returns."""
    obs = envs.reset()

    all_ep_rets = []
    all_ep_rets_norm = []
    all_ep_lens = []

    while len(all_ep_rets) < envs.num_envs:
        with torch.inference_mode():
            obs = torch.from_numpy(obs).permute((0, 3, 1, 2)).to(config.DEVICE)

            action_logits = policy(normalize_obs(obs))
            if hasattr(policy, "decoder"):
                action_logits = policy.decoder(action_logits)

            action = Categorical(logits=action_logits).sample().cpu().numpy()

        obs, _, _, info = envs.step(action)

        for _, item in enumerate(info):
            if "episode" in item.keys():
                # Got this from the PPO code
                all_ep_rets.append(item["episode"]["r"])
                all_ep_rets_norm.append(envs.normalize_return(item["episode"]["r"]))
                all_ep_lens.append(item["episode"]["l"])

    return np.mean(all_ep_rets), np.mean(all_ep_rets_norm), np.mean(all_ep_lens)


def rollout_metrics(step, env_name, gamma, policy, num_envs, logger):
    policy.eval()
    envs = env_utils.setup_procgen_env(
        num_envs=num_envs,
        env_id=env_name,
        gamma=gamma,
    )
    ret, ret_norm, ep_len = rollout(policy, envs)
    logger(
        step=step,
        episodic_return=ret,
        episodic_return_norm=ret_norm,
        episodic_length=ep_len,
    )
