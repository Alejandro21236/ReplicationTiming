#!/bin/bash
#SBATCH --job-name=cellpatchx_job
#SBATCH --output=logs/cellpatchx_%A_%a.out
#SBATCH --error=logs/cellpatchx_%A_%a.err
#SBATCH --nodes=1
#SBATCH --time=150:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300gb
#SBATCH --array=0-19

module load cuda/11.8.0
module load miniconda3/24.1.2-py310
conda activate rnaseq

PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
SCRIPT=${SCRIPT:-"$PROJECT_ROOT/AlexsBullshit.py"}
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

configs=(
  "--use_spatial_bias --use_patch_embeddings --patch_dim 1536 --encoder univ2 --projection_dim 2 --code pancan_2class_withext_spatial_proj2"
  "--use_spatial_bias --use_patch_embeddings --patch_dim 1536 --encoder univ2 --projection_dim 1 --code pancan_2class_withext_spatial_proj1"
  "--use_patch_embeddings --patch_dim 1536 --encoder univ2 --projection_dim 1 --code pancan_2class_withext_proj1"
  "--use_patch_embeddings --patch_dim 1536 --encoder univ2 --projection_dim 2 --code pancan_2class_withext_proj2"
)

config_idx=$((SLURM_ARRAY_TASK_ID / 5))
fold_idx=$((SLURM_ARRAY_TASK_ID % 5))
config="${configs[$config_idx]}"

python "$SCRIPT" $config --fold_idx "$fold_idx" --resume

