#!/bin/bash
#SBATCH --job-name=gp_model    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=msantos@whoi.edu     # Where to send mail
#SBATCH --array=1-10                   # how many tasks in the array
#SBATCH --mem=30gb                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/gp_model_%j_%a.log   # Standard output and error log
pwd; hostname; date

eval "$(conda shell.bash hook)"

conda activate gpytorch

echo "training model_spectral"

cd /vortexfs1/scratch/msantos/gp_plankton_model/

srun python /vortexfs1/scratch/msantos/gp_plankton_model/train/train.py $SLURM_ARRAY_TASK_ID
echo "training finished"


date
