#!/bin/bash -l

#SBATCH -J ledger_probe
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:15:00

set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"; cd "${REPO_DIR}"
mkdir -p logs runs data
source slurm/_env.sh

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "=== git ==="
git branch --show-current || true
git log --oneline -1 || true
echo "=== python ==="
python - <<'PY'
import sys
print("python", sys.version.split()[0], sys.executable)
for mod in ["torch", "torchvision", "numpy", "matplotlib", "wandb"]:
    try:
        m = __import__(mod)
        print(f"{mod:12s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod:12s} MISSING  {repr(e)[:80]}")
import torch
print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
echo "=== files ==="
ls -l scripts/cluster_job.py 2>&1 | head -1
ls -d data/stl10_binary data/cifar-10-batches-py 2>&1 | head -4
echo "=== network ==="
python - <<'PY'
import urllib.request, socket
socket.setdefaulttimeout(20)
for name, url in [("stanford", "http://cs.stanford.edu/~acoates/stl10/"),
                  ("toronto", "https://www.cs.toronto.edu/~kriz/"),
                  ("pypi", "https://pypi.org/simple/")]:
    try:
        urllib.request.urlopen(url).read(256)
        print(f"{name:10s} REACHABLE")
    except Exception as e:
        print(f"{name:10s} BLOCKED  {repr(e)[:90]}")
PY
echo "=== quick job check ==="
python scripts/cluster_job.py --part theorem --out runs/probe \
    --thm_dims 4 --thm_batch 64 --thm_reps 1 --thm_steps 200 --wandb 0
echo "probe done"
