from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .metrics import evaluate_full, retrieval_metrics, two_way_identification
from .diagnostics import dispersion, effective_rank
from .sphere import polar_encode


@dataclass
class FrontierPoint:
    k: int
    strength: float
    subject: int
    n: int
    latent: dict[str, float] = field(default_factory=dict)
    image: dict[str, float] = field(default_factory=dict)

    def flat(self) -> dict:
        row = {"k": self.k, "strength": self.strength,
               "subject": self.subject, "n": self.n}
        row |= {f"latent/{k}": v for k, v in self.latent.items()}
        row |= {f"image/{k}": v for k, v in self.image.items()}
        return row


def parse_ks(spec) -> list[int]:
    if isinstance(spec, str):
        spec = spec.replace(",", " ").split()
    ks = []
    for token in spec:
        if str(token).lower() in ("anchor", "mu", "0"):
            ks.append(0)
        else:
            ks.append(int(token))
    return sorted({k for k in ks})


@torch.no_grad()
def latent_frontier(model, loader, stats, device, *, ks, n_steps=20, solver="heun",
                    max_batches=None, retrieval_pool=300):
    core = model.module if hasattr(model, "module") else model
    was_training = core.training
    core.eval()

    sample_ks = [k for k in ks if k >= 1] or [1]
    banks: dict[int, dict[str, list]] = {k: {"pred": [], "targ": [], "res": []}
                                         for k in ks}
    radius_pred, radius_true = [], []
    seen = 0

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        fmri = batch["fmri"].to(device, non_blocking=True)
        tgt = batch["emb"].to(device, non_blocking=True).float()
        subject = batch["subject"]

        got = core.predict_tokens_frontier(fmri, subject, stats, ks=sample_ks,
                                           n_steps=n_steps, solver=solver)
        z1, r1 = polar_encode(tgt, core.tgt_mean.float())
        for k in ks:
            entry = got["anchor" if k == 0 else k]
            zdir, _ = polar_encode(entry["tokens"].float(), core.tgt_mean.float())
            banks[k]["pred"].append(zdir.to("cpu", torch.float16))
            banks[k]["targ"].append(z1.to("cpu", torch.float16))
            banks[k]["res"].append(entry["resultant"].cpu())

        brain = core.backbone(fmri, subject)
        _, _, logr = core.anchor(brain)
        radius_pred.append(logr.flatten().cpu())
        radius_true.append(r1.log().flatten().cpu())
        seen += fmri.shape[0]

    if was_training:
        core.train()
    if seen == 0:
        return []

    rp = torch.cat(radius_pred).float(); rt = torch.cat(radius_true).float()
    rp = rp - rp.mean(); rt = rt - rt.mean()
    radius_corr = float((rp * rt).sum() / (rp.norm() * rt.norm()).clamp_min(1e-8))

    points = []
    for k in ks:
        P = torch.cat(banks[k]["pred"]).float()
        T = torch.cat(banks[k]["targ"]).float()
        r = retrieval_metrics(P, T, batch_size=retrieval_pool,
                              device=torch.device("cpu"))
        cos = F.cosine_similarity(P, T, dim=-1).mean(dim=-1)
        points.append(FrontierPoint(
            k=k, strength=float("nan"), subject=-1, n=seen,
            latent={
                "token_cos": float(cos.mean()),
                "token_cos_std": float(cos.std()),
                "two_way": two_way_identification(P.flatten(1), T.flatten(1)),
                "retr_fwd": r["retrieval_fwd"],
                "retr_bwd": r["retrieval_bwd"],
                "resultant": float(torch.cat(banks[k]["res"]).mean()),
                "eff_rank": effective_rank(P.flatten(1)),
                "dispersion": dispersion(P.flatten(1)),
                "radius_corr": radius_corr,
            }))
    return points


def predicted_resultant(anchor_cos: float, k: int) -> float:
    m = max(0.0, min(1.0, float(anchor_cos)))
    return math.sqrt(m * m + (1.0 - m * m) / max(1, int(k)))


def predicted_frontier(anchor_cos: float, ks) -> dict[int, float]:
    m = max(0.0, min(1.0, float(anchor_cos)))
    return {int(k): (m * m) / predicted_resultant(m, int(k))
            for k in ks if int(k) >= 1}


@torch.no_grad()
def decode_frontier(model, loader, stats, decoder, device, *, ks, strengths,
                    n_steps=20, solver="heun", max_images=None, subjects=None,
                    full_metrics=True, retrieval_pool=300, on_point=None):
    core = model.module if hasattr(model, "module") else model
    core.eval()
    sample_ks = [k for k in ks if k >= 1] or [1]
    wanted = set(subjects) if subjects else None

    cache: dict[int, dict[int, dict[str, list]]] = {}
    gts: dict[int, list] = {}
    counts: dict[int, int] = {}

    for batch in loader:
        s = int(batch["subject"])
        if wanted is not None and s not in wanted:
            continue
        if max_images is not None and counts.get(s, 0) >= max_images:
            if wanted is not None and all(counts.get(x, 0) >= max_images for x in wanted):
                break
            continue
        take = batch["fmri"].shape[0]
        if max_images is not None:
            take = min(take, max_images - counts.get(s, 0))
        fmri = batch["fmri"][:take].to(device, non_blocking=True)

        got = core.predict_tokens_frontier(fmri, s, stats, ks=sample_ks,
                                           n_steps=n_steps, solver=solver)
        lat = core.predict_low_latent(fmri, s)
        blur = core.predict_lowlevel(fmri, s)

        slot = cache.setdefault(s, {k: {"tok": [], "cls": [], "lat": [], "blur": [],
                                        "emb": []} for k in ks})
        for k in ks:
            entry = got["anchor" if k == 0 else k]
            slot[k]["tok"].append(entry["tokens"].to("cpu", torch.float16))
            slot[k]["cls"].append(None if entry["cls"] is None
                                  else entry["cls"].to("cpu", torch.float16))
            slot[k]["lat"].append(None if lat is None else lat.to("cpu", torch.float16))
            slot[k]["blur"].append(None if blur is None
                                   else blur.to("cpu", torch.float16))
            slot[k]["emb"].append(batch["emb"][:take].to("cpu", torch.float16))
        gts.setdefault(s, []).append(batch["image"][:take])
        counts[s] = counts.get(s, 0) + take

    points = []
    for s, per_k in cache.items():
        gt = torch.cat(gts[s])
        for k in ks:
            tok = torch.cat(per_k[k]["tok"])
            cls_parts = per_k[k]["cls"]
            cls = None if cls_parts[0] is None else torch.cat(cls_parts)
            lat_parts = per_k[k]["lat"]
            lat_all = None if lat_parts[0] is None else torch.cat(lat_parts)
            blur_parts = per_k[k]["blur"]
            blur_all = None if blur_parts[0] is None else torch.cat(blur_parts)
            emb = torch.cat(per_k[k]["emb"])
            for strength in strengths:
                preds = []
                for i in range(0, tok.shape[0], 16):
                    sl = slice(i, i + 16)
                    preds.append(decoder.decode(
                        tok[sl].to(device).float(),
                        None if cls is None else cls[sl].to(device).float(),
                        init_image=(None if blur_all is None or strength >= 1.0
                                    else blur_all[sl].to(device).float()),
                        init_latent=(None if lat_all is None or strength >= 1.0
                                     else lat_all[sl].to(device).float()),
                        strength=strength))
                pred = torch.cat(preds)
                image = evaluate_full(pred, gt, device) if full_metrics else {
                    "PixCorr": float(F.cosine_similarity(
                        pred.flatten(1) - pred.flatten(1).mean(1, keepdim=True),
                        gt.flatten(1) - gt.flatten(1).mean(1, keepdim=True),
                        dim=-1).mean())}
                r = retrieval_metrics(tok.float(), emb.float(),
                                      batch_size=retrieval_pool, device=device)
                image["retrieval_fwd"] = r["retrieval_fwd"]
                image["retrieval_bwd"] = r["retrieval_bwd"]
                pt = FrontierPoint(k=k, strength=float(strength), subject=s,
                                   n=int(tok.shape[0]), image=image)
                points.append(pt)
                if on_point is not None:
                    on_point(pt)
    return points


def pareto_front(points, x="image/PixCorr", y="image/CLIP_2way"):
    rows = [p.flat() for p in points]
    rows = [r for r in rows if x in r and y in r
            and not math.isnan(r[x]) and not math.isnan(r[y])]
    front = []
    for r in rows:
        if not any(o is not r and o[x] >= r[x] and o[y] >= r[y]
                   and (o[x] > r[x] or o[y] > r[y]) for o in rows):
            front.append(r)
    return sorted(front, key=lambda r: r[x])


def format_surface(points, value="image/PixCorr", second="image/CLIP_2way"):
    rows = [p.flat() for p in points]
    ks = sorted({r["k"] for r in rows})
    ss = sorted({r["strength"] for r in rows if not math.isnan(r["strength"])})
    if not ss:
        return ""
    head = "    K\\s |" + "".join(f"{s:>15.2f}" for s in ss)
    lines = [head, "-" * len(head)]
    for k in ks:
        label = "anchor" if k == 0 else str(k)
        cells = []
        for s in ss:
            hit = [r for r in rows if r["k"] == k and r["strength"] == s]
            if not hit:
                cells.append(f"{'':>15}")
            else:
                a = hit[0].get(value, float("nan"))
                b = hit[0].get(second, float("nan"))
                cells.append(f"{a:>7.3f}/{b:<7.3f}")
        lines.append(f"{label:>7} |" + "".join(cells))
    lines.append(f"cells: {value} / {second}")
    return "\n".join(lines)
