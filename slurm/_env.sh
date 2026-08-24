module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0; do
    module load "$m" 2>/dev/null && { echo "module: $m"; break; }
done
for m in anaconda3 anaconda miniconda3 conda; do
    module load "$m" 2>/dev/null && { echo "module: $m"; break; }
done

CONDA_SH=""
for p in "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "$HOME/miniconda3/etc/profile.d/conda.sh" \
         "$HOME/miniforge3/etc/profile.d/conda.sh" \
         "$HOME/mambaforge/etc/profile.d/conda.sh" \
         "/opt/conda/etc/profile.d/conda.sh"; do
    [ -f "$p" ] && { CONDA_SH="$p"; break; }
done
if [ -z "$CONDA_SH" ] && command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
[ -f "$CONDA_SH" ] || { echo "FATAL: no conda.sh found. set CONDA_SH manually."; exit 1; }
source "$CONDA_SH"

ENV_NAME="${ENV_NAME:-me2}"
conda activate "$ENV_NAME" || { echo "FATAL: conda activate $ENV_NAME failed"; conda env list; exit 1; }
python -c "import torch" 2>/dev/null || { echo "FATAL: torch missing in $ENV_NAME ($(which python))"; exit 1; }

echo "env=$CONDA_DEFAULT_ENV python=$(which python)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS=1
