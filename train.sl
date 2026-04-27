#!/usr/bin/bash
# Slurm job script for the UNC CS GPU cluster (gpu.cs.unc.edu)
# Submit with:  sbatch train.sl
# or to train one agent only:  sbatch --export=AGENT=neurosymbolic train.sl

#SBATCH --job-name=nsai-highway
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=/home/pganguli/nsai-highway/results/slurm-%j.out
#SBATCH --error=/home/pganguli/nsai-highway/results/slurm-%j.err

# ── environment ───────────────────────────────────────────────────────────────
PROJECT=/home/pganguli/nsai-highway
PYTHON=${PROJECT}/.venv/bin/python

cd "${PROJECT}"
mkdir -p results  # ensure output dir exists before slurm writes logs

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "=== Job ${SLURM_JOB_ID} started $(date) ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Python: $(${PYTHON} --version)"

# ── training ──────────────────────────────────────────────────────────────────
# AGENT defaults to "all"; override with --export=AGENT=neural or =neurosymbolic
${PYTHON} train_all.py \
    --agent "${AGENT:-all}" \
    --timesteps "${TIMESTEPS:-100000}" \
    ${EXTRA_ARGS:-}

echo "=== Job ${SLURM_JOB_ID} finished $(date) ==="
