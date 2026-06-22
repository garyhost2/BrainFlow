"""Download MindEye2's SDXL-unCLIP decoder checkpoint from the NSD dataset repo.

    python scripts/download_unclip6.py --out third_party
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "pscotti/mindeyev2"
FILENAME = "unclip6_epoch0_step110000.ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="third_party")
    ap.add_argument("--repo", type=str, default=REPO)
    ap.add_argument("--filename", type=str, default=FILENAME)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"▶ Downloading {args.filename} from {args.repo} (dataset)…")
    path = hf_hub_download(
        repo_id=args.repo, filename=args.filename, repo_type="dataset",
        revision="main", local_dir=args.out,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"✓ saved -> {path}")


if __name__ == "__main__":
    main()
