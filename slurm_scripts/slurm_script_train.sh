#!/bin/bash
#SBATCH --job-name=gp_model    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=msantos@whoi.edu     # Where to send mail
#SBATCH --ntasks=10                    # Run on a single CPU
#SBATCH --mem=30gb                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/gp_model_%j.log   # Standard output and error log
pwd; hostname; date

eval "$(conda shell.bash hook)"

conda activate gpytorch

echo "running test job"

cd /vortexfs1/scratch/msantos/gp_plankton_model/

python /vortexfs1/scratch/msantos/gp_plankton_model/train/train.py

date
