#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh

conda init
conda activate gpytorch
echo "activated"
wandb sweep --project gp_sweep cfg/temp_sweep_local.yaml 2> syn_temp.file
cat syn_temp.file
eval "$(awk 'NR==4 {print $6, $7, $8}' syn_temp.file)"
echo "ran wandb sweep"
