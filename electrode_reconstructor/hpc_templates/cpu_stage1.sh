#!/bin/bash

#SBATCH --account=slchen
#SBATCH --partition=pub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o logs/cpu-%x-%j.out
#SBATCH -e logs/cpu-%x-%j.err

SECONDS=0
SHARED_DATASET_ROOT="/scratch/zhushijie/Structure_Data"
LOCAL_ROOT="${SLURM_TMPDIR:-/tmp/${USER}/slurm_${SLURM_JOB_ID}}"
LOCAL_DATASET_ROOT="${LOCAL_ROOT}/Structure_Data"

export MPLCONFIGDIR="${LOCAL_ROOT}/mpl"
export XDG_CACHE_HOME="${LOCAL_ROOT}/xdg-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$LOCAL_DATASET_ROOT"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$LOCAL_DATASET_ROOT/train" "$LOCAL_DATASET_ROOT/verify" "$LOCAL_DATASET_ROOT/test"
[ -d "$SHARED_DATASET_ROOT/train" ] && rsync -a --include='*/' --exclude='*' "$SHARED_DATASET_ROOT/train/" "$LOCAL_DATASET_ROOT/train/"
[ -d "$SHARED_DATASET_ROOT/verify" ] && rsync -a --include='*/' --exclude='*' "$SHARED_DATASET_ROOT/verify/" "$LOCAL_DATASET_ROOT/verify/"
[ -d "$SHARED_DATASET_ROOT/test" ] && rsync -a --include='*/' --exclude='*' "$SHARED_DATASET_ROOT/test/" "$LOCAL_DATASET_ROOT/test/"

python -u electrode-kit.py \
  --pipeline-stage cpu \
  --dataset-root "$LOCAL_DATASET_ROOT" \
  --train-count 6321 \
  --verify-count 380 \
  --test-count 389 \
  --batch-seed 20260428 \
  --epochs 0 \
  </dev/null

mkdir -p "$SHARED_DATASET_ROOT"
rsync -a "$LOCAL_DATASET_ROOT/" "$SHARED_DATASET_ROOT/"
printf 'Total elapsed: %02dh:%02dm:%02ds\n' "$((SECONDS/3600))" "$(((SECONDS%3600)/60))" "$((SECONDS%60))"
