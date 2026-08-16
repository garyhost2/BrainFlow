#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# Gate 0 — is the fMRI <-> image pairing correct?
#
#   sbatch slurm/gate0_offset.sbatch
#
# brainflow/data.py indexes betas with (all_trial - 1) while indexing the image
# hdf5 unshifted. MindEye2 defines behav column 5 as
#     global_trial = (SESSION-1)*750 + jj      # 0-based
# and consumes it with NO offset. If ours is wrong, every arm of gate1 measures
# a scrambled pairing, so gate1_rxfm.sbatch refuses to run until this job has
# written a CONCLUSIVE verdict.
#
# CPU-only work (a ridge fit on held-out trials), but it still has to run under
# SLURM: the login node has system python 3.8 and no env, so `python -m
# scripts.check_offset` there dies on `import h5py`.
#
# --gres=gpu:1 is requested only because every partition on this cluster is a
# GPU partition. If CPU-only jobs are accepted on gpu-all, drop the line.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J bf_gate0
#SBATCH -o logs/gate0_%j.out
#SBATCH -e logs/gate0_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-00:45:00

set -euo pipefail

module load slurm 2>/dev/null || true

# Must be the SAME env as gate1_rxfm.sbatch, which this job gates. That is the
# `brainflow` conda env, not the older `me2` venv the step1b-era scripts use:
# me2 was built by setup_step1b.sh, which installs only what the step1 training
# path needs, and the step1 path reads the tensor cache rather than raw NSD. So
# check_offset -- which does read raw NSD, through h5py and webdataset -- cannot
# run there, while every training job is unaffected.
ENV_NAME="${ENV_NAME:-brainflow}"
[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ] && source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"; cd "${REPO_DIR}"
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }
export TOKENIZERS_PARALLELISM=false

mkdir -p outputs logs
SUBJECT="${SUBJECT:-1}"
N_EVAL="${N_EVAL:-2000}"
OUT_FILE="${OUT_FILE:-outputs/gate0_offset.txt}"

echo "Node: $(hostname)   python: $(command -v python)"
echo "── gate 0: subject ${SUBJECT}, n-eval ${N_EVAL} → ${OUT_FILE}"

# tee through a pipe would hide a python failure behind tee's exit status, and
# `set -o pipefail` would then kill the job before the verdict is written.
python -u -m scripts.check_offset \
    --subject "${SUBJECT}" \
    --n-eval "${N_EVAL}" > "${OUT_FILE}.tmp"
mv "${OUT_FILE}.tmp" "${OUT_FILE}"

cat "${OUT_FILE}"
echo ""
echo "── verdict line gate1 will read:"
grep -E "GATE0:" "${OUT_FILE}" || {
    echo "NO GATE0 LINE — scripts/check_offset.py is older than the gate1 preflight."
    exit 1
}
