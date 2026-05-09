from pathlib import Path

MAX_DATA_CHUNKS = 80

storage_path = Path(".")
_expert_data_path = storage_path / "expert_data"
_experiment_results_path = storage_path / "exp_results"

assert (
    _expert_data_path.exists()
), f"Expert data dir: {_expert_data_path} does not exist"


def get_expert_data(env_name: str, test: bool) -> list[Path]:
    test_flag = "test" if test else "train"
    task_data_path = _expert_data_path / env_name / test_flag
    return sorted(task_data_path.iterdir(), key=lambda x: int(x.stem))[:MAX_DATA_CHUNKS]


def get_experiment_dir(exp_name):
    d = _experiment_results_path / exp_name
    d.mkdir(exist_ok=True, parents=True)
    return d


def get_bc_stage1_path(exp_name, obs, seed):
    return get_experiment_dir(exp_name) / f"bc_stage1_policy_obs{obs}_seed{seed}.pt"


def get_idm_stage1_path(exp_name, obs, seed):
    return get_experiment_dir(exp_name) / f"idm_stage1_idm_obs{obs}_seed{seed}.pt"


def get_idm_stage2_path(exp_name, obs, seed):
    return get_experiment_dir(exp_name) / f"idm_stage2_policy_obs{obs}_seed{seed}.pt"


def get_lapo_stage1_path(exp_name: str):
    return get_experiment_dir(exp_name) / "lapo_stage1_idm_fdm.pt"


def get_lapo_stage2_path(exp_name):
    return get_experiment_dir(exp_name) / "lapo_stage2_latent_policy.pt"


def get_lapo_stage3_path(exp_name, obs, seed, freeze_backbone):
    return get_experiment_dir(exp_name) / f"lapo_stage3_policy_obs{obs}_seed{seed}{'_freeze_backbone' if freeze_backbone else ''}.pt"


def get_lapo_plus_stage2_path(exp_name, obs, seed, freeze_backbone):
    return get_experiment_dir(exp_name) / f"lapo_plus_stage2_idm_obs{obs}_seed{seed}{'_freeze_backbone' if freeze_backbone else ''}.pt"


def get_lapo_plus_stage3_path(exp_name, obs, seed, freeze_backbone):
    return (
        get_experiment_dir(exp_name)
        / f"lapo_plus_stage3_policy_obs{obs}_seed{seed}{'_freeze_backbone' if freeze_backbone else ''}.pt"
    )
