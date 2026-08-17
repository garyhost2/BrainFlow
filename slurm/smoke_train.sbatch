#!/bin/bash -l

#SBATCH -J step1b_smoke
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 02:00:00

set -euo pipefail
module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0; do
    module load "$m" 2>/dev/null && { echo "Loaded module: $m"; break; }
done

VENV="${VENV:-$HOME/envs/rxfm}"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate "${ENV_NAME:-rxfm}"
fi

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"; cd "${REPO_DIR}"
echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -m scripts.smoke_decoder --subject 1 --n 8 \
    --mindeye-src third_party/MindEyeV2/src \
    --ckpt-path third_party/unclip6_epoch0_step110000.ckpt \
    --out outputs/step1b/smoke

echo "Done. Inspect outputs/step1b/smoke/smoke_grid.png"
