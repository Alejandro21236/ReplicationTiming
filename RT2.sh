#!/bin/bash
#SBATCH --account=
#SBATCH --job-name=vit_rt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=vit.out
#SBATCH --error=vit.err

set -euo pipefail

# ---- PATHS ----
WORKDIR="/fs/scratch/PAS2942/Alejandro/RT" 
SCRIPT="RT3.py"
PATCHES_DIR="/fs/scratch/PAS2942/TCGA_DS_1/20x/BRCA/patches"
LABELS_CSV="/fs/scratch/PAS2942/Alejandro/datasets/20x_RT.csv"
OUTROOT="/fs/scratch/PAS2942/Alejandro/RT/RTreduced2"

# ---- ENV ----
module load cuda/11.8.0 || true
export CUDA_VISIBLE_DEVICES=0
# source ~/.bashrc && conda activate YOUR_ENV

mkdir -p "${OUTROOT}"
cd "${WORKDIR}"

# ---- TRAIN ----
srun -u python "${SCRIPT}" \
  --patches_dir "${PATCHES_DIR}" \
  --labels_csv "${LABELS_CSV}" \
  --out_dir "${OUTROOT}" \
  --epochs 6 \
  --folds 5 \
  --batch_size 1 \
  --max_patches 64 \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --freeze_blocks 9 \
  --seed 42 \
  --target_cols "rt_mean"
echo "All done."
