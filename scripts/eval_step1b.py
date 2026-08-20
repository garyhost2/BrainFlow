from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from brainflow.tensor_cache import assert_tensor_cache_alignment
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
    ap.add_argument("--no-decode", action="store_true",
                    help="embedding-space metrics only: skip SDXL entirely")
    ap.add_argument("--blend-w", type=float, default=None,
                    help="cond_source=blend: 1.0 = full flow sample, 0.0 = anchor")
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
    ap.add_argument("--n-samples", type=int, default=1,
                    help="K in the multi-sample estimator: average K independent "
                         "transports and renormalise. K=1 is a posterior sample, "
                         "large K approaches the posterior mean.")
    ap.add_argument("--max-images", type=int, default=982,
                    help="images PER SUBJECT (982 = the full NSD test set)")
    ap.add_argument("--eval-subjects", type=int, nargs="+", default=None,
                    help="which of the checkpoint's subjects to score. Default: the "
                         "first one only (~1.2 h). Every subject is scored SEPARATELY "
                         "and a mean row is reported -- pooling them would put the same "
                         "image in another subject's foil set.")
    ap.add_argument("--no-full-metrics", dest="full_metrics", action="store_false",
                    default=True,
                    help="skip the 8-metric NSD table (AlexNet/Inception/EffNet/SwAV)")
    ap.add_argument("--no-retrieval", dest="retrieval", action="store_false", default=True,
                    help="skip image/brain retrieval (needs the predicted tokens in RAM)")
    ap.add_argument("--retrieval-pool", type=int, default=300,
                    help="candidates per retrieval pool (NSD convention: 300)")
    ap.add_argument("--out", type=str, default="outputs/step1b/eval")
    ap.add_argument("--moment-match", action="store_true",
                    help="rescale the predicted img2img latent to unit per-channel "
                         "variance before unstandardising. An L1 head shrinks toward "
                         "the mean, so its latent has too little variance and the VAE "
                         "decodes it pale and low-contrast; this restores the second "
                         "moment. Cannot add information -- sweep it, do not assume it.")
    ap.add_argument("--wandb-project", type=str, default=None,
                    help="W&B project. Unset disables tracking. Logs the metric "
                         "table and the reconstruction grids as images.")
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-name", type=str, default=None,
                    help="run name; defaults to the basename of --out")
    ap.add_argument("--wandb-mode", type=str, default="online",
                    choices=["online", "offline"])
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
    if args.blend_w is not None:
        cfg.blend_w = args.blend_w
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

    tensors = assert_tensor_cache_alignment(data_dir / args.tensor_cache, torch.load(data_dir / args.tensor_cache, map_location="cpu"))

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

    if args.no_decode and not args.retrieval:
        raise SystemExit("--no-decode with --no-retrieval measures nothing")
    if args.no_decode:
        decoder = clip_metric = None
        print("▶ --no-decode: retrieval only, SDXL not loaded")
    else:
        decoder = SDXLUnCLIPDecoder(device, args.mindeye_src, args.ckpt_path,
                                    num_steps=args.decode_steps,
                                    cls_vector_slot=tuple(args.cls_vector_slot))
        clip_metric = CLIPMetric(device, hf_cache=hf_cache)

    want = args.eval_subjects or [subjects[0]]
    unknown = [s for s in want if s not in subjects]
    if unknown:
        raise SystemExit(f"--eval-subjects {unknown} not in the checkpoint ({subjects})")
    print(f"▶ scoring subjects {want} separately, ≤{args.max_images} images each "
          f"(~1.2 h per subject at 982). Checkpoint covers {subjects}.")

    # Collect per subject. Scoring subjects separately is not just cosmetic: 2-way
    # identification and retrieval draw their foils from the scored set, so pooling
    # subjects would put the SAME image (seen by someone else) in the foil pool and
    # deflate both. It is also the only way the numbers line up with the
    # single-subject rows in the published tables.
    bank: dict[int, dict[str, list]] = {s: {"pred": [], "gt": [], "tp": [], "tg": []}
                                        for s in want}
    with torch.no_grad():
        for batch in tqdm(bundle.eval, desc="eval"):
            s = int(batch["subject"])
            if s not in bank:
                continue
            got = (bank[s].get("n", 0) if args.no_decode
                   else sum(p.shape[0] for p in bank[s]["pred"]))
            if got >= args.max_images:
                if all(sum(p.shape[0] for p in b["pred"]) >= args.max_images
                       for b in bank.values()):
                    break
                continue
            take = min(batch["fmri"].shape[0], args.max_images - got)
            fmri = batch["fmri"][:take].to(device, non_blocking=True)
            tok, cls_hat = model.predict_tokens(fmri, s, stats,
                                                n_samples=args.n_samples)
            if not args.no_decode:
                blur = model.predict_lowlevel(fmri, s)
                lat = model.predict_low_latent(fmri, s, moment_match=args.moment_match)
                bank[s]["pred"].append(decoder.decode(tok, cls_hat, init_image=blur,
                                                      init_latent=lat,
                                                      strength=cfg.ll_strength))
                bank[s]["gt"].append(batch["image"][:take])
            else:
                bank[s]["n"] = bank[s].get("n", 0) + int(tok.shape[0])
            if args.retrieval:
                # Retrieval runs on the predicted embedding, not on pixels. Park it
                # on CPU in fp16: 982 x 256 x 1664 is ~0.8 GB per side in fp16.
                bank[s]["tp"].append(tok.detach().to("cpu", torch.float16))
                bank[s]["tg"].append(batch["emb"][:take].to("cpu", torch.float16))

    def _score_retrieval_only(n, tp, tg):
        m = {"n": int(n)}
        if tp:
            m.update(retrieval_metrics(torch.cat(tp), torch.cat(tg),
                                       batch_size=args.retrieval_pool, device=device))
        return m

    def _score(pred, gt, tp, tg):
        # Legacy keys keep the pre-2026-08 definitions so every historical run
        # stays comparable; the unprefixed keys are the NSD-convention ones.
        m = {"n": int(pred.shape[0]),
             "legacy_PixCorr": pixcorr(pred, gt), "legacy_SSIM": ssim(pred, gt)}
        m.update({f"legacy_{k}": v for k, v in clip_metric.score(pred, gt).items()})
        if args.full_metrics:
            m.update(evaluate_full(pred, gt, device))
        else:
            m["PixCorr"] = m["legacy_PixCorr"]; m["SSIM"] = m["legacy_SSIM"]
            m["CLIP_2way"] = m["legacy_CLIP_2way"]
        if tp:
            m.update(retrieval_metrics(torch.cat(tp), torch.cat(tg),
                                       batch_size=args.retrieval_pool, device=device))
        return m

    per_subject = {}
    for s in want:
        b = bank[s]
        if args.no_decode:
            if not b["tp"]:
                print(f"[warn] subject {s}: no batches collected, skipping")
                continue
            per_subject[s] = _score_retrieval_only(b.get("n", 0), b["tp"], b["tg"])
            r = per_subject[s]
            print(f"  subj{s:02d}: fwd={r.get('retrieval_fwd', float('nan')):.4f} "
                  f"bwd={r.get('retrieval_bwd', float('nan')):.4f} "
                  f"bwd-fwd={r.get('retrieval_bwd', float('nan')) - r.get('retrieval_fwd', float('nan')):+.4f} "
                  f"(n={r['n']})", flush=True)
            b["tp"].clear(); b["tg"].clear()
            continue
        if not b["pred"]:
            print(f"[warn] subject {s}: no batches collected, skipping")
            continue
        pred = torch.cat(b["pred"]); gt = torch.cat(b["gt"])
        per_subject[s] = _score(pred, gt, b["tp"], b["tg"])
        print(f"  subj{s:02d}: PixCorr={per_subject[s]['PixCorr']:.3f} "
              f"SSIM={per_subject[s]['SSIM']:.3f} "
              f"CLIP_2way={per_subject[s]['CLIP_2way']:.3f} (n={per_subject[s]['n']})",
              flush=True)
        try:
            from torchvision.utils import save_image
            k = min(16, pred.shape[0])
            save_image(torch.cat([gt[:k], pred[:k]]),
                       str(out_dir / f"recon_grid_s{s}.png"), nrow=k)
        except Exception as e:
            print(f"[grid skipped] {e}")
        b["pred"].clear(); b["gt"].clear(); b["tp"].clear(); b["tg"].clear()

    if not per_subject:
        raise SystemExit("no subject produced any images")

    # Mean over subjects, which is how the published tables aggregate. Averaged
    # only over subjects that actually reported a key (SwAV can be skipped if the
    # torch.hub download fails), so one missing metric cannot silently bias it.
    keys = [k for k, v in next(iter(per_subject.values())).items()
            if isinstance(v, (int, float))]
    mean = {}
    for k in keys:
        vals = [v[k] for v in per_subject.values() if k in v]
        if vals:
            mean[k] = float(sum(vals) / len(vals))
    out = {"config": {"cond_source": args.cond_source, "cfg_scale": args.cfg_scale,
                      "steps": args.steps, "solver": args.solver,
                      "n_samples": args.n_samples, "blend_w": cfg.blend_w,
                      "ll_strength": cfg.ll_strength, "ckpt": args.ckpt},
           "subjects": sorted(per_subject), "per_subject": per_subject, "mean": mean}
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    _wandb_publish(args, out, out_dir)
    print("\n" + json.dumps(mean, indent=2))
    if args.full_metrics:
        label = (f"mean of {len(per_subject)} subjects" if len(per_subject) > 1
                 else f"subject {sorted(per_subject)[0]}")
        print(f"\n=== {label} vs the published NSD table ===")
        print(format_comparison(mean))
    print(f"\n✓ metrics -> {out_dir/'metrics.json'}")
    print(f"✓ grids   -> {out_dir}/recon_grid_s*.png")

def _wandb_publish(args, out, out_dir):
    """Mirror the metric table and the recon grids to W&B. Never fatal: a
    tracking failure must not lose an eval that already wrote metrics.json."""
    if not args.wandb_project:
        return
    try:
        import os
        import wandb
    except ImportError:
        print("[warn] --wandb-project set but wandb is not installed")
        return
    mode = args.wandb_mode
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        print("[warn] WANDB_API_KEY unset; logging offline")
        mode = "offline"
    try:
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name or Path(out_dir).name,
                         mode=mode, dir=str(out_dir), job_type="eval",
                         config=out["config"])
        flat = {f"eval/{k}": v for k, v in out["mean"].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for subj, m in out.get("per_subject", {}).items():
            flat.update({f"eval/s{subj}/{k}": v for k, v in m.items()
                         if isinstance(v, (int, float)) and not isinstance(v, bool)})
        run.log(flat)
        run.summary.update(flat)
        grids = sorted(Path(out_dir).glob("recon_grid_s*.png"))
        if grids:
            run.log({"reconstructions": [
                wandb.Image(str(g),
                            caption=f"{g.stem}: top row ground truth, bottom row reconstruction")
                for g in grids]})
        print(f"✓ W&B {mode}: {getattr(run, 'url', out_dir)}")
        run.finish()
    except Exception as e:
        print(f"[warn] wandb publish failed ({e}); metrics.json is unaffected")


if __name__ == "__main__":
    main()
