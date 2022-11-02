#!/bin/bash
#SBATCH --job-name=train_gp_model    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=msantos@whoi.edu     # Where to send mail
#SBATCH --ntasks=5                 # how many tasks in the array
#SBATCH --mem-per-cpu=20gb                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/train_%a_%j.log   # Standard output and error log
pwd; hostname; date

eval "$(conda shell.bash hook)"

conda activate gpytorch

echo "training model_spectral"

cd /vortexfs1/scratch/msantos/gp_plankton_model/

srun --ntasks=5 -l --multi-prog slurm_scripts/slurm_config.conf
echo "training finished"


date
