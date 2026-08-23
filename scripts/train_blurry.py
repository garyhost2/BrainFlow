"""Train MindEye2's low-level pathway on our data, scored by their metric.

    python -m scripts.train_blurry --subjects 1 2 5 7 --epochs 30 \
        --out outputs/blurry_mindeye

The number that decides it is `pixcorr` on held-out images. Their own bar, from
Train.ipynb, is **0.456** -- what MindEye v1's low-level branch scored on
subject 1. Our seven attempts were measured against blur_mse < 1.0 and never
got there; this is a sharper target and a known-achievable one.

Standalone by design: it never touches the token encoder, so it cannot degrade
retrieval or the semantic metrics. Its output is a blurry RGB image handed to
decoder_sgm's `init_image`, which already exists.

Target latents are encoded once and kept on the GPU (~340 MB per subject);
re-encoding 27k images through the VAE every epoch would dominate the run.
ConvNeXt runs live because the augmented view changes each step.
"""
import argparse
import json
import pathlib
import time

import torch

from brainflow.denoise import RepeatIndex
from brainflow.step1.blurry_mindeye import (BLURRY_PIXCORR_BAR, MindEyeBlurry,
                                            blurry_loss, blurry_pixcorr,
                                            build_blur_augs, encode_target,
                                            load_autoenc, load_convnext)
from brainflow.tensor_cache import assert_tensor_cache_alignment


def _images01(imgs_u8, sel, device, size=224):
    """[0,1] images at MindEye2's resolution.

    Our cache stores 256x256 (config.img_size); their whole low-level path is
    224. The SD-VAE downsamples by 8, so 256 gives a 4x32x32 latent while
    blin1 emits 3136 = 4x28x28 -- the first run died on exactly that mismatch.
    ConvNeXt and their RandomResizedCrop are 224 as well, so one resize here
    keeps the target, the auxiliary loss and the augmentation consistent.
    """
    x = imgs_u8[sel.cpu()].to(device, non_blocking=True)
    x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
    if x.shape[-1] != size:
        x = torch.nn.functional.interpolate(x, size, mode="bilinear",
                                            align_corners=False).clamp(0, 1)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-cache", type=str,
                    default="./mindeyev2_cache/all_subjects_tensors.pt")
    ap.add_argument("--mindeye-src", type=str, default="third_party/MindEyeV2/src")
    ap.add_argument("--autoenc", type=str,
                    default="./mindeyev2_cache/sd_image_var_autoenc.pth")
    ap.add_argument("--convnext", type=str,
                    default="./mindeyev2_cache/convnext_xlarge_alpha0.75_fullckpt.pth")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--h", type=int, default=4096)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--blur-scale", type=float, default=0.5)
    ap.add_argument("--cont-weight", type=float, default=0.1)
    ap.add_argument("--no-cont", action="store_true",
                    help="ablate the ConvNeXt term -- the ingredient none of our "
                         "seven attempts had")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--image-size", type=int, default=224,
                    help="MindEye2's low-level path is 224 end to end; our cache "
                         "is 256, and 256/8=32 does not fit blin1's 4x28x28")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/blurry_mindeye")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"▶ MindEye2 blurry pathway | subjects {args.subjects} | {device}", flush=True)
    autoenc = load_autoenc(args.autoenc, device)
    cnx = None if args.no_cont else load_convnext(args.convnext, args.mindeye_src, device)
    augs = build_blur_augs()
    if cnx is not None and augs is None:
        print("[warn] kornia unavailable: the augmented view falls back to the "
              "image itself, which degenerates the contrastive term", flush=True)
    print(f"✓ autoencoder loaded | convnext {'off' if cnx is None else 'on'} "
          f"| augs {'on' if augs is not None else 'off'}", flush=True)

    p = pathlib.Path(args.tensor_cache)
    blob = assert_tensor_cache_alignment(
        str(p), torch.load(p, map_location="cpu", mmap=True))

    subj = {}
    for s in args.subjects:
        fmri = blob[f"fmri_train_{s}"].float()
        st = blob["fmri_stats"][s]
        x = ((fmri - st["mu"].float()) / st["std"].float().clamp_min(1e-6)).to(device)
        imgs = blob[f"imgs_train_{s}"]
        idx = RepeatIndex.cached(imgs, p.parent / f"repeat_index_s{s}.pt")
        tr, va = idx.split(args.val_frac, seed=args.seed)

        # Encode every target once. 27k x 4x28x28 fp32 is ~340 MB.
        lat = torch.empty(len(imgs), 4, 28, 28, device=device)
        for i0 in range(0, len(imgs), 64):
            sel = torch.arange(i0, min(i0 + 64, len(imgs)))
            lat[i0:i0 + len(sel)] = encode_target(_images01(imgs, sel, device, args.image_size), autoenc)
        subj[s] = {"x": x, "imgs": imgs, "lat": lat,
                   "train": tr.to(device), "val": va.to(device),
                   "n_vox": x.shape[1]}
        print(f"  subj{s:02d}: {x.shape[0]} trials, {x.shape[1]} voxels, "
              f"{len(tr)} train / {len(va)} val, targets encoded", flush=True)

    model = MindEyeBlurry({s: d["n_vox"] for s, d in subj.items()},
                          args.mindeye_src, h=args.h, n_blocks=args.n_blocks,
                          drop=args.drop).to(device)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"✓ {n_par/1e6:.1f}M trainable parameters", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    counts = torch.tensor([len(subj[s]["train"]) for s in subj], dtype=torch.float)
    order = list(subj)
    steps = max(1, int(counts.sum().item()) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)

    @torch.no_grad()
    def evaluate():
        model.eval()
        res = {}
        for s, d in subj.items():
            tot, n = 0.0, 0
            for i0 in range(0, min(len(d["val"]), args.eval_batches * args.batch_size),
                            args.batch_size):
                sel = d["val"][i0:i0 + args.batch_size]
                pred, _ = model(d["x"][sel], s)
                img = _images01(d["imgs"], sel, device, args.image_size)
                tot += blurry_pixcorr(pred, img, autoenc) * len(sel)
                n += len(sel)
            res[s] = tot / max(n, 1)
        model.train()
        return res

    best, hist = -1e9, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        agg = {"l1": 0.0, "cont": 0.0, "n": 0}
        for _ in range(steps):
            s = order[int(torch.multinomial(counts, 1).item())]
            d = subj[s]
            sel = d["train"][torch.randint(len(d["train"]), (args.batch_size,),
                                           device=device)]
            pred, aux = model(d["x"][sel], s)
            img = _images01(d["imgs"], sel, device, args.image_size)
            ld = blurry_loss(pred, aux, img, d["lat"][sel], cnx, augs,
                             args.cont_weight)
            loss = ld["loss"] * args.blur_scale
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            agg["l1"] += ld["l1"].item()
            agg["cont"] += float(ld.get("cont", 0.0))
            agg["n"] += 1

        ev = evaluate()
        mean_pc = sum(ev.values()) / len(ev)
        row = {"epoch": ep, "l1": agg["l1"] / agg["n"], "cont": agg["cont"] / agg["n"],
               "pixcorr": mean_pc, "per_subject": {str(k): v for k, v in ev.items()},
               "secs": round(time.time() - t0, 1)}
        hist.append(row)
        with open(out_dir / "blurry_log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

        mark = ""
        if mean_pc > best:
            best = mean_pc
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "voxels": {s: d["n_vox"] for s, d in subj.items()},
                        "epoch": ep, "pixcorr": mean_pc}, out_dir / "best.pt")
            mark = "  ✓best.pt"
        if mean_pc > BLURRY_PIXCORR_BAR:
            mark += f"  ★ over MindEye's {BLURRY_PIXCORR_BAR} bar"
        detail = " ".join(f"s{k}:{v:.3f}" for k, v in sorted(ev.items()))
        print(f"Ep {ep:3d} | l1={row['l1']:.4f} cont={row['cont']:.4f} | "
              f"pixcorr={mean_pc:.4f} | {detail} | {row['secs']}s{mark}", flush=True)

    print(f"\nDone. best blurry pixcorr = {best:.4f} "
          f"(MindEye2's bar is {BLURRY_PIXCORR_BAR})")
    if best <= BLURRY_PIXCORR_BAR:
        print("Below the bar. Since this is THEIR architecture, THEIR loss and "
              "THEIR target, a miss points at the data path or the training "
              "setup, not at the low-level idea -- which is what seven "
              "from-scratch attempts could never distinguish.")


if __name__ == "__main__":
    main()
