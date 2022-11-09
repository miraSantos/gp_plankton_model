conda activate gpytorch

wandb sweep --project syn_model_sweep cfg/sweep_config.yaml 2> syn_temp.file

for i in {1...4}; do
  eval "$(awk 'NR==4 {print $6, $7, $8}' syn_temp.file)"
  done