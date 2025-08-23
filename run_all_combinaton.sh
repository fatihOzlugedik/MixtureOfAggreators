#!/bin/bash
#
#SBATCH --job-name=moa_router_full
#SBATCH --output=logs_moa_ablation/moa_%A_%a.out
#SBATCH --error=logs_moa_ablation/moa_%A_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --array=21-125   # adjust range as needed

# Safer shell flags: turn on -u AFTER sourcing env files
set -eo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# 1) Environment (avoid unbound var errors while sourcing)
# ──────────────────────────────────────────────────────────────────────────────
mkdir -p logs_moa_ablation

set +u
# system and user profiles (ignore if missing)
source /etc/profile 2>/dev/null || true
source "$HOME/.bashrc" 2>/dev/null || true
set -u

# activate conda (guard if not present)
if command -v conda >/dev/null 2>&1; then
  set +u
  conda activate caitomorph 2>/dev/null || true
  set -u
fi

cd /lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/mixture_of_aggregators
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# ──────────────────────────────────────────────────────────────────────────────
# 2) Search space
# ──────────────────────────────────────────────────────────────────────────────
expert_modes=(shared separate shared_adapter)
router_styles=(dense topk)
router_types=(linear mlp transformer)
use_local_head=(0 1)
num_experts=(2 4)
# topk will be derived per (router_style, num_expert)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Build only meaningful combos into a flat array
# ──────────────────────────────────────────────────────────────────────────────
declare -a COMBOS=()

for mode in "${expert_modes[@]}"; do
  for rstyle in "${router_styles[@]}"; do
    for rtype in "${router_types[@]}"; do
      for lh in "${use_local_head[@]}"; do
        for ne in "${num_experts[@]}"; do
          if [[ "$rstyle" == "dense" ]]; then
            k=1
            COMBOS+=("$mode $rstyle $rtype $lh $ne $k")
          else
            # topk: only k <= num_expert (for ne=2 -> k=1,2 ; for ne=4 -> k=1,2,3)
            maxk=$(( ne == 2 ? 2 : 3 ))
            for k in $(seq 1 $maxk); do
              COMBOS+=("$mode $rstyle $rtype $lh $ne $k")
            done
          fi
        done
      done
    done
  done
done

TOTAL=${#COMBOS[@]}
echo "[INFO] Total meaningful combos = $TOTAL (expect 126 if full range)"

# Map SLURM index
IDX=${SLURM_ARRAY_TASK_ID}
if (( IDX < 0 || IDX >= TOTAL )); then
  echo "[FATAL] Index $IDX out of range 0..$((TOTAL-1))"
  exit 1
fi

read -r EXPERT_MODE ROUTER_STYLE ROUTER_TYPE USE_LOCAL_HEAD NUM_EXPERT TOPK <<< "${COMBOS[$IDX]}"

# ──────────────────────────────────────────────────────────────────────────────
# 4) Fixed / common args
# ──────────────────────────────────────────────────────────────────────────────
ARCH="MixtureOfAggregators"
DATA_PATH="/lustre/groups/labs/marr/qscd01/workspace/beluga_features_extracted/dinobloom-b"
SAVING_NAME="Beluga_full"
SEED=38

# ──────────────────────────────────────────────────────────────────────────────
# 5) Echo combo to logs
# ──────────────────────────────────────────────────────────────────────────────
echo "[INFO] Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "       arch=$ARCH mode=$EXPERT_MODE rstyle=$ROUTER_STYLE rtype=$ROUTER_TYPE lh=$USE_LOCAL_HEAD numexp=$NUM_EXPERT k=$TOPK"

# ──────────────────────────────────────────────────────────────────────────────
# 6) Launch
# ──────────────────────────────────────────────────────────────────────────────
python -u train_5fold_test_fixed.py \
  --arch "$ARCH" \
  --data_path "$DATA_PATH" \
  --saving_name "$SAVING_NAME" \
  --seed "$SEED" \
  --expert_mode "$EXPERT_MODE" \
  --router_style "$ROUTER_STYLE" \
  --router_type "$ROUTER_TYPE" \
  --num_expert "$NUM_EXPERT" \
  --topk "$TOPK" \
  $( [[ "$USE_LOCAL_HEAD" -eq 1 ]] && echo "--use_local_head" ) \
  --save_gates \
  > "logs_moa_ablation/moa_${ARCH}_mode${EXPERT_MODE}_rstyle${ROUTER_STYLE}_rtype${ROUTER_TYPE}_lh${USE_LOCAL_HEAD}_ne${NUM_EXPERT}_k${TOPK}.txt"
