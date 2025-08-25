#!/bin/bash

#SBATCH -o logs_baseline/slurm_pyscript_%j_test_fixed.job
#SBATCH -e logs_baseline/slurm_error_%j_test_fixed.job
#SBATCH -p gpu_p
#SBATCH --qos=gpu_priority
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00              # shorter walltime for quick smoke test
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --nice=10000

source $HOME/.bashrc
cd /lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/mixture_of_aggregators
export PYTHONPATH=/lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/mixture_of_aggregators:$PYTHONPATH
mkdir logs_baseline
conda activate caitomorph

# run a single fold with fewer epochs just to test
python -u train_5fold_test_fixed.py \
    --arch Transformer \
    --data_path /lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/BRACS/uni_features \
    --saving_name Bracs_baseline
  
    > logs_baseline/output_debug_run.txt
