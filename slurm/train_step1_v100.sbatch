#!/bin/bash -l

#SBATCH -J step1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100nv_32GB:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 1-00:00:00

set -euo pipefail

module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0 cuda12.4/toolkit/12.4.1; do
    module load "$m" 2>/dev/null && { echo "Loaded module: $m"; break; }
done

ENV_NAME="${ENV_NAME:-brainflow}"
[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ] && source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"
cd "${REPO_DIR}"
mkdir -p logs outputs
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }

python -c "import transformers" 2>/dev/null || pip install --quiet transformers

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "Node: $(hostname)"; nvidia-smi || true

python -m scripts.train_step1 \
    --data-dir ./mindeyev2_cache \
    --subjects 1 \
    --epochs 150 --batch-size 256 --lr 3e-4 \
    --eval-freq 5 --decode-eval --decode-n 64 \
    --out outputs/step1

echo "Done at $(date)"
