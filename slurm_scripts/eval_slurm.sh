#!/bin/bash
#SBATCH --job-name=eval_gp_model    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=msantos@whoi.edu     # Where to send mail
#SBATCH --array=6-9                    # Run on a single CPU
#SBATCH --mem=30gb                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/eval_%a_%j.log   # Standard output and error log
pwd; hostname; date

eval "$(conda shell.bash hook)"

conda activate gpytorch

echo "running evaluation"

cd /vortexfs1/scratch/msantos/gp_plankton_model/

srun python /vortexfs1/scratch/msantos/gp_plankton_model/evaluation/evaluate.py $SLURM_ARRAY_TASK_ID
echo "evaluation finished"

date
