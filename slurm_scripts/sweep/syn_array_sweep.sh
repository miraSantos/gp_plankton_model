#!/bin/bash
#SBATCH --job-name=array_sweep_gp_model    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=msantos@whoi.edu     # Where to send mail
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=/vortexfs1/scratch/msantos/gp_plankton_model/slurm_scripts/sweep/logs/syn_array_sweep.log   # Standard output and error log
pwd; hostname; date

eval "$(conda shell.bash hook)"

conda activate gpytorch

echo "training model_spectral"a

cd /vortexfs1/scratch/msantos/gp_plankton_model

wandb sweep --project syn_model_sweep cfg/sweep_syn_config.yaml 2> temp.file

cat temp.file

for i in {1}; do
  eval "$(awk 'NR==4 {print $6, $7, $8}' temp.file)" &
  done
wait
echo "sweep finished"

date
