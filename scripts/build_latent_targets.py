from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from rxfm.tensor_cache import assert_tensor_cache_alignment
from rxfm.decoder import SDXLUnCLIPDecoder, quiet_benign_warnings
from rxfm.targets_latent import build_or_load_latent_targets, _subj_file


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--tensor-cache", type=str, default="all_subjects_tensors.pt")
    ap.add_argument("--target-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--mindeye-src", type=str, default="third_party/MindEyeV2/src")
    ap.add_argument("--ckpt-path", type=str,
                    default="third_party/unclip6_epoch0_step110000.ckpt")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--ll-size", type=int, default=64,
                    help="blur level: GT is downsampled to this before encoding")
    ap.add_argument("--force", action="store_true", help="rebuild existing caches")
    return ap.parse_args()


def main():
    args = parse_args()
    quiet_benign_warnings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dir = Path(args.target_dir)

    todo = [s for s in args.subjects
            if args.force or not _subj_file(target_dir, s, args.ll_size).exists()]
    if not todo:
        print("✓ every requested cache already exists; nothing to do.")
        return

    need_gb = 2.2 * len(todo)
    free_gb = shutil.disk_usage(target_dir).free / 1e9
    print(f"── disk preflight: to-build={len(todo)} subject(s) ≈{need_gb:.1f}G, "
          f"free={free_gb:.1f}G on {target_dir}")
    if free_gb < need_gb + 20:
        raise SystemExit(
            f"ERROR: need ~{need_gb:.1f}G (+20G headroom) but only {free_gb:.1f}G free.")

    tensors = assert_tensor_cache_alignment(Path(args.data_dir) / args.tensor_cache, torch.load(Path(args.data_dir) / args.tensor_cache, map_location="cpu"))
    print("▶ loading SDXL-unCLIP (for its VAE)…")
    decoder = SDXLUnCLIPDecoder(device, args.mindeye_src, args.ckpt_path)

    out = build_or_load_latent_targets(tensors, args.subjects, target_dir, decoder,
                                       ll_size=args.ll_size, force_rebuild=args.force)
    st = out["_lat_stats"]
    print(f"\n✓ done. latent channel mean={st['mean'].tolist()}")
    print(f"        latent channel std ={st['std'].tolist()}")
    for s in args.subjects:
        fp = _subj_file(target_dir, s, args.ll_size)
        print(f"   subj{s:02d}: {fp} ({fp.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
