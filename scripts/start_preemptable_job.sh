#!/bin/bash 
PARTITION=gpu-2080ti-preemptable
#FLAGS= 
PYTHONPATH=. srun --job-name="$JOB_NAME" --partition=$PARTITION --cpus-per-task=2 --mem=8G --gres=gpu:1 -- /mnt/qb/work/bethge/preizinger/ima-vae//scripts/run_singularity_server.sh python3 /mnt/qb/work/bethge/preizinger/ima-vae/ima_vae/simulations.py  --project mlp-test  --use-wandb --n-steps 250001 "$@"

