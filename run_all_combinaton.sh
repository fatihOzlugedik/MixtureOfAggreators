#!/bin/bash
#
#SBATCH --job-name=moa_router_full
#SBATCH --output=logs_AML_Hehr_expert_choose/moa_%A_%a.out
#SBATCH --error=logs_AML_Hehr_expert_choose/moa_%A_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --array=0-0

set -eo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# 1) Environment
# ──────────────────────────────────────────────────────────────────────────────
mkdir -p logs_AML_Hehr_expert_choose

set +u
source /etc/profile 2>/dev/null || true
source "$HOME/.bashrc" 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate caitomorph 2>/dev/null || true
fi
set -u

cd /lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/mixture_of_aggregators
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# ──────────────────────────────────────────────────────────────────────────────
# 2) Search space
# ──────────────────────────────────────────────────────────────────────────────
expert_modes=(separate)
router_styles=(topk)
router_types=(mlp transformer linear)
use_local_head=(0)
num_experts=(10)

# Make top_k a simple independent dimension (edit values as you like)
topk_list=(2)

# Routing toggles
lb_toggle=(0 1)
gumbel_toggle=(0 1)

# Default params (used when toggles are on)
LB_COEF_DEFAULT=0.01
GUMBEL_TAU_START=2.0
GUMBEL_TAU_MIN=0.5
GUMBEL_TAU_DECAY=0.95

# ──────────────────────────────────────────────────────────────────────────────
# 3) Build combos (no heuristics, no k>ne checks)
# ──────────────────────────────────────────────────────────────────────────────
declare -a COMBOS=()
for mode in "${expert_modes[@]}"; do
  for rstyle in "${router_styles[@]}"; do
    for rtype in "${router_types[@]}"; do
      for lh in "${use_local_head[@]}"; do
        for ne in "${num_experts[@]}"; do
          for k in "${topk_list[@]}"; do
            for lb in "${lb_toggle[@]}"; do
              for gum in "${gumbel_toggle[@]}"; do
                COMBOS+=("$mode $rstyle $rtype $lh $ne $k $lb $gum")
              done
            done
          done
        done
      done
    done
  done
done

TOTAL=${#COMBOS[@]}
echo "[INFO] Total combos = $TOTAL"

# Use 1-based SLURM array indexing
IDX=$((SLURM_ARRAY_TASK_ID - 1))
if (( IDX < 0 || IDX >= TOTAL )); then
  echo "[SKIP] Index $SLURM_ARRAY_TASK_ID (0-based $IDX) out of range 1..$TOTAL"
  exit 0
fi

read -r EXPERT_MODE ROUTER_STYLE ROUTER_TYPE USE_LOCAL_HEAD NUM_EXPERT TOPK USE_LB_TOGGLE USE_GUMBEL_TOGGLE <<< "${COMBOS[$IDX]}"

# ──────────────────────────────────────────────────────────────────────────────
# 4) Fixed/common args
# ──────────────────────────────────────────────────────────────────────────────
ARCH="MixtureOfAggregators"
DATA_PATH="/lustre/groups/labs/marr/qscd01/workspace/AML_Hehr_features_extracted/dinobloom-b"
SAVING_NAME="AML_Hehr_expert_count_choose"
CSV_ROOT="/lustre/groups/labs/marr/qscd01/workspace/AML_Hehr_features_extracted/dinobloom-b/folds"
LABEL_MAP_CSV="/lustre/groups/labs/marr/qscd01/workspace/AML_Hehr_features_extracted/dinobloom-b/folds/label_map.csv"
SEED=38

BACKBONE="$(basename "$DATA_PATH")"
LOCALHEAD_STR=$([[ "$USE_LOCAL_HEAD" -eq 1 ]] && echo "True" || echo "False")
LB_STR=$([[ "$USE_LB_TOGGLE" -eq 1 ]] && echo "on" || echo "off")
GUMBEL_STR=$([[ "$USE_GUMBEL_TOGGLE" -eq 1 ]] && echo "on" || echo "off")

SAVING_NAME_TOGGLED="${SAVING_NAME}/gumbel_${GUMBEL_STR}_lb_${LB_STR}"

RESULT_FOLDER_ROOT="Results_5fold_testfixed_${BACKBONE}_${ARCH}_${EXPERT_MODE}_${ROUTER_STYLE}_topk${TOPK}_localhead${LOCALHEAD_STR}_router_arch_${ROUTER_TYPE}_seed${SEED}"
FOLD4_DIR="${SAVING_NAME_TOGGLED}/${NUM_EXPERT}_experts/${RESULT_FOLDER_ROOT}/fold_4"
FOLD4_PNG="${FOLD4_DIR}/confusion_matrix.png"
FOLD4_NPY="${FOLD4_DIR}/test_conf_matrix.npy"

# ──────────────────────────────────────────────────────────────────────────────
# 5) Echo combo
# ──────────────────────────────────────────────────────────────────────────────
echo "[INFO] Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "       arch=$ARCH mode=$EXPERT_MODE rstyle=$ROUTER_STYLE rtype=$ROUTER_TYPE lh=$USE_LOCAL_HEAD numexp=$NUM_EXPERT k=$TOPK lb=${USE_LB_TOGGLE} gumbel=${USE_GUMBEL_TOGGLE}"
echo "       expecting results under: ${FOLD4_DIR}"

# ──────────────────────────────────────────────────────────────────────────────
# 6) Pre-flight skip if fold_4 already completed
# ──────────────────────────────────────────────────────────────────────────────
if [[ -f "$FOLD4_PNG" || -f "$FOLD4_NPY" ]]; then
  echo "[SKIP] Found existing fold_4 results for this combo:"
  [[ -f "$FOLD4_PNG" ]] && echo "       - $FOLD4_PNG"
  [[ -f "$FOLD4_NPY" ]] && echo "       - $FOLD4_NPY"
  exit 0
fi

# ──────────────────────────────────────────────────────────────────────────────
# 7) Launch
# ──────────────────────────────────────────────────────────────────────────────
python -u train_5fold_test_fixed.py \
  --arch "$ARCH" \
  --data_path "$DATA_PATH" \
  --saving_name "$SAVING_NAME_TOGGLED" \
  --seed "$SEED" \
  --expert_mode "$EXPERT_MODE" \
  --router_style "$ROUTER_STYLE" \
  --router_type "$ROUTER_TYPE" \
  --num_expert "$NUM_EXPERT" \
  --topk "$TOPK" \
  $( [[ "$USE_LOCAL_HEAD" -eq 1 ]] && echo "--use_local_head" ) \
  $( [[ "$USE_LB_TOGGLE" -eq 1 ]] && echo "--use_lb_loss --lb_coef ${LB_COEF_DEFAULT}" ) \
  $( [[ "$USE_GUMBEL_TOGGLE" -eq 1 ]] && echo "--use_gumbel --gumbel_tau_start ${GUMBEL_TAU_START} --gumbel_tau_min ${GUMBEL_TAU_MIN} --gumbel_decay ${GUMBEL_TAU_DECAY}" ) \
  --save_gates \
  --csv_root "$CSV_ROOT" \
  --label_map_csv "$LABEL_MAP_CSV" \
  | tee "logs_AML_Hehr_expert_choose/moa_${ARCH}_mode${EXPERT_MODE}_rstyle${ROUTER_STYLE}_rtype${ROUTER_TYPE}_lh${USE_LOCAL_HEAD}_ne${NUM_EXPERT}_k${TOPK}_lb${USE_LB_TOGGLE}_g${USE_GUMBEL_TOGGLE}.txt"
