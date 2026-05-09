#!/bin/bash

# Cluster-specific example — adjust paths/partitions for your environment.

# Parameters
#SBATCH --job-name=custom_job
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mail-type=ARRAY_TASKS,FAIL,TIME_LIMIT
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=main-cpu
#SBATCH -o slurm-%j.out

LAPO_DIR=${LAPO_DIR:-$HOME/vla_wm/LAPO}
LAPO_VENV=${LAPO_VENV:-$HOME/venvs/lapo}

cd $LAPO_DIR
module load python/3.10
source $LAPO_VENV/bin/activate
bash setup_data.sh