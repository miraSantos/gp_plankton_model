#!/bin/bash
#SBATCH --job-name=gp_model    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=msantos@whoi.edu     # Where to send mail
#SBATCH --ntasks=10                    # Run on a single CPU
#SBATCH --mem=30gb                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/gp_model_%j.log   # Standard output and error log
pwd; hostname; date

module load anaconda3/2021.11

source activate gp

module load python3/3.10.2

echo "running test job"

python ../train/train.py

date