from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from rxfm.tensor_cache import assert_tensor_cache_alignment
from rxfm.model import TokenStep1Config, TokenStep1Model
from rxfm.target_stats import TargetStats
from rxfm.targets import build_or_load_bigg_targets
from rxfm.dataset import build_step1_loaders
from rxfm.decoder import SDXLUnCLIPDecoder, quiet_benign_warnings
from rxfm.image_metrics import pixcorr, ssim, CLIPMetric
from rxfm.metrics import evaluate_full, retrieval_metrics, format_comparison

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

    bank: dict[int, dict[str, list]] = {s: {"pred": [], "gt": [], "tp": [], "tg": []}
                                        for s in want}
    with torch.no_grad():
        for batch in tqdm(bundle.eval, desc="eval"):
            s = int(batch["subject"])
            if s not in bank:
                continue
            got = sum(p.shape[0] for p in bank[s]["pred"])
            if got >= args.max_images:
                if all(sum(p.shape[0] for p in b["pred"]) >= args.max_images
                       for b in bank.values()):
                    break
                continue
            take = min(batch["fmri"].shape[0], args.max_images - got)
            fmri = batch["fmri"][:take].to(device, non_blocking=True)
            tok, cls_hat = model.predict_tokens(fmri, s, stats,
                                                n_samples=args.n_samples)
            blur = model.predict_lowlevel(fmri, s)
            lat = model.predict_low_latent(fmri, s)
            bank[s]["pred"].append(decoder.decode(tok, cls_hat, init_image=blur,
                                                  init_latent=lat,
                                                  strength=cfg.ll_strength))
            bank[s]["gt"].append(batch["image"][:take])
            if args.retrieval:
                bank[s]["tp"].append(tok.detach().to("cpu", torch.float16))
                bank[s]["tg"].append(batch["emb"][:take].to("cpu", torch.float16))

    def _score(pred, gt, tp, tg):
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

    keys = [k for k, v in next(iter(per_subject.values())).items()
            if isinstance(v, (int, float))]
    mean = {}
    for k in keys:
        vals = [v[k] for v in per_subject.values() if k in v]
        if vals:
            mean[k] = float(sum(vals) / len(vals))
    out = {"config": {"cond_source": args.cond_source, "cfg_scale": args.cfg_scale,
                      "steps": args.steps, "solver": args.solver,
                      "n_samples": args.n_samples,
                      "ll_strength": cfg.ll_strength, "ckpt": args.ckpt},
           "subjects": sorted(per_subject), "per_subject": per_subject, "mean": mean}
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    print("\n" + json.dumps(mean, indent=2))
    if args.full_metrics:
        label = (f"mean of {len(per_subject)} subjects" if len(per_subject) > 1
                 else f"subject {sorted(per_subject)[0]}")
        print(f"\n=== {label} vs the published NSD table ===")
        print(format_comparison(mean))
    print(f"\n✓ metrics -> {out_dir/'metrics.json'}")
    print(f"✓ grids   -> {out_dir}/recon_grid_s*.png")

if __name__ == "__main__":
    main()
