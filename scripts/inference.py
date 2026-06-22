from __future__ import annotations
import os, sys, gc, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brainflow.config import load_config
from brainflow.data import build_dataloaders
from brainflow.models import BrainFlowV5, migrate_input_proj
from brainflow.vae import FrozenVAE
from brainflow.metrics import pixel_correlation, ssim_pytorch

load_dotenv()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Path to .pt; defaults to outputs/best_pc_v5.pt")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--out-prefix", type=str, default="inference_v5")
    ap.add_argument("--cfg-scale", type=float, default=None,
                    help="CFG scale for NFE sweep (defaults to cfg.cfg_scale)")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Cap number of batches per sweep (None = full eval set)")
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, eval_loader, _, voxels = build_dataloaders(cfg)
    test_ds = test_loader.dataset

    model = BrainFlowV5(cfg, voxels).to(device).eval()
    ckpt = Path(args.ckpt) if args.ckpt else (cfg.output_dir / "best_pc_v5.pt")
    if ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu")
        sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
        sd = migrate_input_proj(sd, model.brain_enc.max_vox)
        model.load_state_dict(sd)
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print(f"⚠ checkpoint not found at {ckpt}; using random weights")

    vae = FrozenVAE(cache_dir=cfg.data_dir / "hf_cache").to(device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = args.n_samples
    fig, axes = plt.subplots(3, n, figsize=(n * 2.8, 9))
    fig.suptitle(f"BrainFlow v5 — fMRI → Image (CFG={cfg.cfg_scale}, {cfg.ode_steps} steps)",
                 fontsize=13, fontweight="bold")
    pcs = []
    with torch.no_grad():
        for i in range(n):
            s = test_ds[i]
            fmri = s["fmri"].unsqueeze(0).to(device)
            subj = torch.tensor([s["subject"]], device=device)
            gt = s["image"].permute(1, 2, 0).numpy().clip(0, 1)
            tokens, _ = model.encode_fmri(fmri, subj)
            pl = model.sample(tokens)
            pr = vae.decode(pl).squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            pc = pearsonr(gt.flatten(), pr.flatten())[0]; pcs.append(pc)
            axes[0, i].imshow(gt); axes[0, i].axis("off")
            axes[0, i].set_title(f"GT subj{s['subject']} #{i}", fontsize=8)
            axes[1, i].imshow(pr); axes[1, i].axis("off")
            axes[1, i].set_title(f"r={pc:.3f}", fontsize=8,
                                 color="green" if pc > 0.15 else "red")
            axes[2, i].imshow(np.abs(gt - pr), cmap="hot", vmin=0, vmax=0.5)
            axes[2, i].axis("off")

    plt.tight_layout()
    out_png = cfg.output_dir / f"{args.out_prefix}.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"mean PC={np.mean(pcs):.4f} ± {np.std(pcs):.4f} | best={np.max(pcs):.4f}")
    print(f"Saved: {out_png}")

    print("\nCFG sweep:")
    for s in [0.0, 1.0, 1.5, 2.0, 3.0, 5.0]:
        pcs2, sss2 = [], []
        n_samples_total = 0
        t0 = time.time()
        with torch.no_grad():
            for bi, batch in enumerate(eval_loader):
                if args.max_batches is not None and bi >= args.max_batches:
                    break
                fmri = batch["fmri"].to(device)
                subject = batch["subject"].to(device)
                images = batch["image"]
                tokens, _ = model.encode_fmri(fmri, subject)
                pl = model.sample(tokens, n_steps=cfg.ode_steps, cfg_scale=s)
                pi = vae.decode(pl)
                pcs2.append(pixel_correlation(pi, images.to(device)))
                sss2.append(ssim_pytorch(pi.cpu(), images))
                n_samples_total += fmri.shape[0]
        elapsed_ms = (time.time() - t0) * 1000
        ms_per_sample = elapsed_ms / max(1, n_samples_total)
        print(f"  CFG={s:.1f} | PC={np.mean(pcs2):.3f} SSIM={np.mean(sss2):.3f} "
              f"| {ms_per_sample:.1f}ms/sample")

    nfe_cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.cfg_scale
    print(f"\nNFE sweep (cfg_scale={nfe_cfg_scale:.1f}):")
    for nfe in [1, 2, 3, 5, 10, 20, 50]:
        pcs3, sss3 = [], []
        with torch.no_grad():
            for bi, batch in enumerate(eval_loader):
                if args.max_batches is not None and bi >= args.max_batches:
                    break
                fmri = batch["fmri"].to(device)
                subject = batch["subject"].to(device)
                images = batch["image"]
                tokens, _ = model.encode_fmri(fmri, subject)
                pl = model.sample(tokens, n_steps=nfe, cfg_scale=nfe_cfg_scale)
                pi = vae.decode(pl)
                pcs3.append(pixel_correlation(pi, images.to(device)))
                sss3.append(ssim_pytorch(pi.cpu(), images))

        print(f"  NFE={nfe:3d} | PC={np.mean(pcs3):.3f} SSIM={np.mean(sss3):.3f}")

if __name__ == "__main__":
    main()
