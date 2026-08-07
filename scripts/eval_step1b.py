from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from brainflow.step1.model_tokens import TokenStep1Config, TokenStep1Model
from brainflow.step1.targets import TargetStats
from brainflow.step1.targets_bigg import build_or_load_bigg_targets
from brainflow.step1.data import build_step1_loaders
from brainflow.step1.decoder_sgm import SDXLUnCLIPDecoder, quiet_benign_warnings
from brainflow.step1.metrics import pixcorr, ssim, CLIPMetric
from brainflow.metrics_full import evaluate_full, retrieval_metrics, format_comparison

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--tensor-cache", type=str, default="all_subjects_tensors.pt")
    ap.add_argument("--target-dir", type=str, default="./mindeyev2_cache",
                    help="dir holding per-subject bigG target files (step1b_bigg_s{N}.pt)")
    ap.add_argument("--mindeye-src", type=str, default="third_party/MindEyeV2/src")
    ap.add_argument("--ckpt-path", type=str, default="third_party/unclip6_epoch0_step110000.ckpt")
    ap.add_argument("--cond-source", type=str, default="prior",
                    choices=["regression", "prior", "blend"])
    ap.add_argument("--cfg-scale", type=float, default=3.0,
                    help="guidance scale for the patch prior")
    ap.add_argument("--cls-cfg-scale", type=float, default=3.0,
                    help="guidance scale for the global CLS prior")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    ap.add_argument("--decode-steps", type=int, default=38)
    ap.add_argument("--cls-vector-slot", type=int, nargs=2, default=[0, 1280],
                    help="[lo hi] positions in the unCLIP 'vector' slot to fill with c_cls")
    ap.add_argument("--ll-strength", type=float, default=None,
                    help="override the checkpoint's img2img strength (low-level pathway)")
    ap.add_argument("--max-images", type=int, default=10_000)
    ap.add_argument("--no-full-metrics", dest="full_metrics", action="store_false",
                    default=True,
                    help="skip the 8-metric NSD table (AlexNet/Inception/EffNet/SwAV)")
    ap.add_argument("--no-retrieval", dest="retrieval", action="store_false", default=True,
                    help="skip image/brain retrieval (needs the predicted tokens in RAM)")
    ap.add_argument("--retrieval-pool", type=int, default=300,
                    help="candidates per retrieval pool (NSD convention: 300)")
    ap.add_argument("--out", type=str, default="outputs/step1b/eval")
    return ap.parse_args()

def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    quiet_benign_warnings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir); hf_cache = data_dir / "hf_cache"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg: TokenStep1Config = ckpt["cfg"]
    cfg.cond_source = args.cond_source; cfg.cfg_scale = args.cfg_scale
    cfg.cls_cfg_scale = args.cls_cfg_scale
    cfg.n_steps = args.steps; cfg.solver = args.solver
    if args.ll_strength is not None:
        cfg.ll_strength = args.ll_strength
    stats = TargetStats.from_dict(ckpt["stats"])
    subjects = ckpt["subjects"]; voxels = ckpt["voxels"]

    model = TokenStep1Model(cfg, voxels)
    # Non-strict: a prior-only checkpoint (trained before the low head) lacks
    # low_head.* weights -> drop the untrained head so decode is token-only.
    res = model.load_state_dict(ckpt["model"], strict=False)
    if any("low_head" in k for k in res.missing_keys):
        model.low_head = None
        print("  checkpoint has no trained low_head -> token-only decode")
    if "ema" in ckpt:
        sd = model.state_dict()
        for k, v in ckpt["ema"].items():
            if k in sd:
                sd[k].copy_(v)
        model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    tensors = torch.load(data_dir / args.tensor_cache, map_location="cpu")

    targets = None
    legacy = data_dir / "step1b_targets_bigg.pt"
    if legacy.exists():
        blob = torch.load(legacy, map_location="cpu")
        if all(f"emb_train_{s}" in blob and f"emb_test_{s}" in blob for s in subjects):
            print(f"✓ using legacy target cache: {legacy}")
            targets = blob
    if targets is None:
        targets = build_or_load_bigg_targets(
            tensors, subjects, args.target_dir, device, args.mindeye_src, hf_cache=hf_cache)
    bundle = build_step1_loaders(tensors, targets, subjects, batch_size=32, num_workers=8)

    decoder = SDXLUnCLIPDecoder(device, args.mindeye_src, args.ckpt_path,
                                num_steps=args.decode_steps,
                                cls_vector_slot=tuple(args.cls_vector_slot))
    clip_metric = CLIPMetric(device, hf_cache=hf_cache)

    preds, gts = [], []
    tok_pred, tok_gt = [], []
    subjects_seen = []
    with torch.no_grad():
        for batch in tqdm(bundle.eval, desc="eval"):
            fmri = batch["fmri"].to(device, non_blocking=True)
            tok, cls_hat = model.predict_tokens(fmri, batch["subject"], stats)
            blur = model.predict_lowlevel(fmri, batch["subject"])
            lat = model.predict_low_latent(fmri, batch["subject"])
            preds.append(decoder.decode(tok, cls_hat, init_image=blur, init_latent=lat,
                                        strength=cfg.ll_strength))
            gts.append(batch["image"])
            subjects_seen.append(int(batch["subject"]))
            if args.retrieval:
                # Retrieval runs on the predicted embedding, not on pixels. Park it
                # on CPU in fp16: 982 x 256 x 1664 is ~0.8 GB per side in fp16.
                tok_pred.append(tok.detach().to("cpu", torch.float16))
                tok_gt.append(batch["emb"].to("cpu", torch.float16))
            if sum(p.shape[0] for p in preds) >= args.max_images:
                break
    n = min(args.max_images, sum(p.shape[0] for p in preds))
    pred = torch.cat(preds)[:n]; gt = torch.cat(gts)[:n]

    # Legacy keys: identical definitions to every run before 2026-08, kept so the
    # new NSD-convention numbers below can be diffed against the old ones.
    metrics = {"n": pred.shape[0],
               "subjects_evaluated": sorted(set(subjects_seen)),
               "legacy_PixCorr": pixcorr(pred, gt), "legacy_SSIM": ssim(pred, gt)}
    metrics.update({f"legacy_{k}": v for k, v in clip_metric.score(pred, gt).items()})

    if args.full_metrics:
        metrics.update(evaluate_full(pred, gt, device))
    else:
        metrics["PixCorr"] = metrics["legacy_PixCorr"]
        metrics["SSIM"] = metrics["legacy_SSIM"]

    if args.retrieval and tok_pred:
        tp = torch.cat(tok_pred)[:n]; tg = torch.cat(tok_gt)[:n]
        metrics.update(retrieval_metrics(tp, tg, batch_size=args.retrieval_pool,
                                         device=device))
        del tp, tg
    tok_pred.clear(); tok_gt.clear()

    print(json.dumps(metrics, indent=2))
    if len(set(subjects_seen)) > 1:
        print(f"\n[warn] {len(set(subjects_seen))} subjects pooled into one metric set. "
              "2-way / retrieval foils then include the SAME image seen by another "
              "subject, which deflates them. Use --max-images 982 for a single subject.")
    if args.full_metrics:
        print("\n" + format_comparison(metrics))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    try:
        from torchvision.utils import save_image
        k = min(16, pred.shape[0])
        save_image(torch.cat([gt[:k], pred[:k]]), str(out_dir / "recon_grid.png"), nrow=k)
        print(f"✓ grid -> {out_dir/'recon_grid.png'}")
    except Exception as e:
        print(f"[grid skipped] {e}")

if __name__ == "__main__":
    main()
