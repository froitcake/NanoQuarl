#!/bin/bash
# Train Nano Quarl on IBM circuits with auto-discovery
# Uses ppo_nano.py as main training script

set -e

echo "🚀 Nano Quarl - IBM Auto-Discovery Training"
echo "============================================"

# Activate environment
source ~/mambaforge/etc/profile.d/conda.sh
conda activate quartz

# Jetson optimizations
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run training with auto-discovery
echo "Starting training at $(date)"
START=$(date +%s)

python ppo_nano.py -cn nano_ibm_auto

END=$(date +%s)
echo "✅ Completed in $((END - START)) seconds ($((($END - $START) / 60)) minutes)"