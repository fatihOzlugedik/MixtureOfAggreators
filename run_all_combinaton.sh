#!/bin/bash
#
#SBATCH --job-name=moa_router_full
#SBATCH --output=logs_moa_BRACS_3/moa_%A_%a.out
#SBATCH --error=logs_moa_BRACS_3/moa_%A_%a.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --array=0-200

set -eo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# 1) Environment
# ──────────────────────────────────────────────────────────────────────────────
mkdir -p logs_moa_BRACS_3

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
expert_modes=(shared_adapter shared separate)                        # 3
router_styles=(topk)                                                 # 1
router_types=(mlp transformer linear)                                # 3
use_local_head=(0 1)                                                 # 2
num_experts=(2 3 4 5 6 7)                                            # 6
# topk: ne=2 -> k in {1,2}; ne>=3 -> k in {1,2,3}

# ──────────────────────────────────────────────────────────────────────────────
# 3) Build meaningful combos
# ──────────────────────────────────────────────────────────────────────────────
declare -a COMBOS=()
for mode in "${expert_modes[@]}"; do
  for rstyle in "${router_styles[@]}"; do
    for rtype in "${router_types[@]}"; do
      for lh in "${use_local_head[@]}"; do
        for ne in "${num_experts[@]}"; do
          maxk=$(( ne == 2 ? 2 : 3 ))
          for k in $(seq 1 "$maxk"); do
            COMBOS+=("$mode $rstyle $rtype $lh $ne $k")
          done
        done
      done
    done
  done
done

TOTAL=${#COMBOS[@]}
echo "[INFO] Total combos = $TOTAL"

IDX=${SLURM_ARRAY_TASK_ID}
if (( IDX < 0 || IDX >= TOTAL )); then
  echo "[SKIP] Index $IDX out of range 0..$((TOTAL-1))"
  exit 0
fi

read -r EXPERT_MODE ROUTER_STYLE ROUTER_TYPE USE_LOCAL_HEAD NUM_EXPERT TOPK <<< "${COMBOS[$IDX]}"

# ──────────────────────────────────────────────────────────────────────────────
# 4) Fixed/common args
# ──────────────────────────────────────────────────────────────────────────────
ARCH="MixtureOfAggregators"
DATA_PATH="/lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/BRACS/uni_features"
SAVING_NAME="BRACS_3Class"                                  # base save root
CSV_ROOT="/lustre/groups/labs/marr/qscd01/workspace/fatih.oezluegedik/mixture_of_aggregators/data_cross_val_3_classes_Bracs"
LABEL_MAP_CSV="$CSV_ROOT/3class_label_mapping.csv"
SEED=38

BACKBONE="$(basename "$DATA_PATH")"
LOCALHEAD_STR=$([[ "$USE_LOCAL_HEAD" -eq 1 ]] && echo "True" || echo "False")

# Expected result path (must match train script naming)
RESULT_FOLDER_ROOT="Results_5fold_testfixed_${BACKBONE}_${ARCH}_${EXPERT_MODE}_${ROUTER_STYLE}_topk${TOPK}_localhead${LOCALHEAD_STR}_router_arch_${ROUTER_TYPE}_seed${SEED}"
FOLD4_DIR="${SAVING_NAME}/${NUM_EXPERT}_experts/${RESULT_FOLDER_ROOT}/fold_4"
FOLD4_PNG="${FOLD4_DIR}/confusion_matrix.png"
FOLD4_NPY="${FOLD4_DIR}/test_conf_matrix.npy"

# ──────────────────────────────────────────────────────────────────────────────
# 5) Echo combo
# ──────────────────────────────────────────────────────────────────────────────
echo "[INFO] Job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "       arch=$ARCH mode=$EXPERT_MODE rstyle=$ROUTER_STYLE rtype=$ROUTER_TYPE lh=$USE_LOCAL_HEAD numexp=$NUM_EXPERT k=$TOPK"
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
mkdir -p logs_moa_BRACS_3
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
  --csv_root "$CSV_ROOT" \
  --label_map_csv "$LABEL_MAP_CSV" \
  | tee "logs_moa_BRACS_3/moa_${ARCH}_mode${EXPERT_MODE}_rstyle${ROUTER_STYLE}_rtype${ROUTER_TYPE}_lh${USE_LOCAL_HEAD}_ne${NUM_EXPERT}_k${TOPK}.txt"
