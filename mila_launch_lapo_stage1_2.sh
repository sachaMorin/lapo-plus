#!/bin/bash

# Cluster-specific example — adjust paths/partitions for your environment.

# Parameters
#SBATCH --job-name=lapo
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=0-6:00:00
#SBATCH --partition=long
#SBATCH --mail-type=ARRAY_TASKS,FAIL,TIME_LIMIT
#SBATCH -o slurm-%j.out
#SBATCH --array=0-15  # Change this to 0-15 to run all envs

SEED=11

# check if the first argument is provided
if [ -z "$1" ]; then
  echo "Error: Missing argument. You should specifiy a sweep name."
  echo "Usage: $0 <argument>"
  exit 1
fi

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

export WANDB_MODE=online

# Use the SLURM_ARRAY_TASK_ID to select the task
ind=${SLURM_ARRAY_TASK_ID}
sweep_name=$1
exp_name="${ind}_${sweep_name}"

echo "Starting new experiment for env: ${tasks[${ind}]}"

python train_lapo_stage1.py env_name="${tasks[${ind}]}" seed="${SEED}" exp_name="${exp_name}" &&
echo "Completed lapo_stage1 for env: ${tasks[${ind}]}" &&
python train_lapo_stage2.py env_name="${tasks[${ind}]}" seed="${SEED}" exp_name="${exp_name}" &&
echo "Completed lapo_stage2 for env: ${tasks[${ind}]}" &&
echo "All stages completed for env: ${tasks[${ind}]}"
