#!/bin/bash

# Cluster-specific example — adjust paths/partitions for your environment.

# Parameters
#SBATCH --job-name=lapo
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=0-48:00:00
#SBATCH --partition=long
#SBATCH --mail-type=ARRAY_TASKS,FAIL,TIME_LIMIT
#SBATCH -o slurm-%j.out
#SBATCH --array=0-15  # Change this to 0-15 to run all tasks

SWEEP_NAME=$1
FREEZE_BACKBONE=$2 # for lapo_plus_stage2
LR=$3 # for lapo_plus_stage2
STEPS=$4 # for lapo_plus_stage2
IDM_ONLY=$5 # set to true for IDM labeling

# this job takes more time so we launch all seeds as separate jobs
SEED=$6

# check if all arguments are provided
if [ -z "$SWEEP_NAME" ] || [ -z "$SEED" ] || [ -z "$FREEZE_BACKBONE" ] || [ -z "$LR" ]; then
  echo "Error: Missing argument(s). You should specifiy a sweep name, seed, freeze_backbone flag, and learning rate."
  echo "Usage: $0 <sweep_name> <seed> <freeze_backbone> <learning_rate>"
  exit 1
fi

SEEDS=($SEED)
OBSERVED_SAMPLES=(16 32 64 128 256 512 1024 2048 4096 8192 32768 131072 524288 -1)

# setup
LAPO_DIR=${LAPO_DIR:-$HOME/vla_wm/LAPO}
LAPO_VENV=${LAPO_VENV:-$HOME/venvs/lapo}
module load python/3.10
source $LAPO_VENV/bin/activate
cd $LAPO_DIR/lapo

# lapo script
declare -A tasks=(
	[0]="bigfish"
	[1]="bossfight"
	[2]="caveflyer"
	[3]="chaser"
	[4]="climber"
	[5]="coinrun"
	[6]="dodgeball"
	[7]="fruitbot"
	[8]="heist"
	[9]="jumper"
	[10]="leaper"
	[11]="maze"
	[12]="miner"
	[13]="ninja"
	[14]="plunder"
	[15]="starpilot"
)

# Wandb offline and log to /tmp
export WANDB_MODE=offline
export WANDB_DIR=/tmp/wandb

# Use the SLURM_ARRAY_TASK_ID to select the task
ind=${SLURM_ARRAY_TASK_ID}
exp_name="${ind}_${SWEEP_NAME}"

echo "Starting new experiment for env: ${tasks[${ind}]}"

# Loop over seeds and observed samples
for seed in "${SEEDS[@]}"; do
  for obs in "${OBSERVED_SAMPLES[@]}"; do
	# Run the stage2_idm_decoding.py script with the current seed and observed samples
	python train_lapo_plus_stage2.py env_name="${tasks[${ind}]}" exp_name="${exp_name}" seed="${seed}" lapo_plus_stage2.n_observed_samples="${obs}" lapo_plus_stage2.freeze_backbone=${FREEZE_BACKBONE} lapo_plus_stage2.lr=${LR} lapo_plus_stage2.steps=${STEPS} lapo_plus_stage2.idm_only=${IDM_ONLY} &&
	echo "Completed lapo_plus_stage2 for env: ${tasks[${ind}]}, seed: ${seed}, observed samples: ${obs}"
	python train_lapo_plus_stage3.py env_name="${tasks[${ind}]}" exp_name="${exp_name}" seed="${seed}" lapo_plus_stage2.n_observed_samples="${obs}" lapo_plus_stage2.freeze_backbone=${FREEZE_BACKBONE} lapo_plus_stage2.idm_only=${IDM_ONLY} &&
	echo "Completed lapo_plus_stage3 for env: ${tasks[${ind}]}, seed: ${seed}"
  done
done

echo "All tasks completed for env: ${tasks[${ind}]}"

# moving wandb logs to scratch
if [[ "$IDM_ONLY" == "true" ]]; then
    out="idm"
elif [[ "$FREEZE_BACKBONE" == "false" ]]; then
	out="lapo_plus_finetuned"
else
    out="lapo_plus"
fi

mkdir -p $SCRATCH/wandb_logs/${out}/
mv /tmp/wandb $SCRATCH/wandb_logs/${out}/${exp_name}_${SEED}

