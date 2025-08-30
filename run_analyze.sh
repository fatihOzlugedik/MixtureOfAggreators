#!/bin/bash

#SBATCH -o logs/slurm_pyscript_%j_test_fixed.job
#SBATCH -e logs/slurm_error_%j_test_fixed.job
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal

#SBATCH -t 12:00:00              # shorter walltime for quick smoke test
#SBATCH -c 30
#SBATCH --mem=40G
#SBATCH --nice=10000

source $HOME/.bashrc
cd /lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/mixture_of_aggregators
export PYTHONPATH=/lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/mixture_of_aggregators:$PYTHONPATH
conda activate dinoBloom2

# run a single fold with fewer epochs just to test
python analyze_script.py \
--root /lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/mixture_of_aggregators/AML_Hehr_expert_count_choose
