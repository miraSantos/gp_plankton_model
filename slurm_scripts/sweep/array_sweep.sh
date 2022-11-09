#!/bin/bash
#SBATCH --job-name=array_sweep_gp_model    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=msantos@whoi.edu     # Where to send mail
#SBATCH --array=1-5               # how many tasks in the array
#SBATHC --cpus-per-task=1
#SBATCH --mem-per-cpu=20G                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=/vortexfs1/scratch/msantos/gp_plankton_model/slurm_scripts/sweep/logs/array_sweep_%a_%j.log   # Standard output and error log
pwd; hostname; date

eval "$(conda shell.bash hook)"

conda activate gpytorch

echo "training model_spectral"

cd /vortexfs1/scratch/msantos/gp_plankton_model

wandb agent mira_thesis/syn_model_sweep/0fhkf9kl

echo "training finished"

date
