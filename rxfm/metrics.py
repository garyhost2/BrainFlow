from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from typing import Any

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

_IN_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IN_STD  = torch.tensor([0.229, 0.224, 0.225])

_MODEL_CACHE: dict[str, Any] = {}

ALEXNET_EARLY_LAYER = 4
ALEXNET_MID_LAYER = 12

def pixel_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.flatten(1).cpu().numpy()
    t = target.flatten(1).cpu().numpy()
    return float(np.nanmean([pearsonr(pi, ti)[0] for pi, ti in zip(p, t)]))

def ssim_pytorch(pred: torch.Tensor, target: torch.Tensor,
                 ws: int = 11, sigma: float = 1.5) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    B, C, H, W = pred.shape
    g = torch.arange(ws, dtype=torch.float32) - ws // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    k = g.outer(g).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1).to(pred.device)
    pad = ws // 2
    def mu(x):
        return F.conv2d(x, k, padding=pad, groups=C)
    mx = mu(pred); my = mu(target)
    sx = mu(pred * pred) - mx ** 2
    sy = mu(target * target) - my ** 2
    sxy = mu(pred * target) - mx * my
    return float(((2 * mx * my + C1) * (2 * sxy + C2) /
                  ((mx ** 2 + my ** 2 + C1) * (sx + sy + C2))).mean())

_GRAY_W = torch.tensor([0.2125, 0.7154, 0.0721])

def ssim_grayscale(pred: torch.Tensor, target: torch.Tensor,
                   ws: int = 11, sigma: float = 1.5) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    w = _GRAY_W.to(pred.device, torch.float32).view(1, 3, 1, 1)
    p = (pred.float() * w).sum(1, keepdim=True)
    t = (target.float() * w).sum(1, keepdim=True)
    g = torch.arange(ws, dtype=torch.float32, device=pred.device) - (ws - 1) / 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    k = g.outer(g).view(1, 1, ws, ws)
    def mu(x):
        return F.conv2d(x, k)
    mp = mu(p); mt = mu(t)
    sp = mu(p * p) - mp ** 2
    st = mu(t * t) - mt ** 2
    spt = mu(p * t) - mp * mt
    return float(((2 * mp * mt + C1) * (2 * spt + C2) /
                  ((mp ** 2 + mt ** 2 + C1) * (sp + st + C2))).mean())

def _normalise(imgs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.view(1, 3, 1, 1).to(imgs.device)
    std  = std.view(1, 3, 1, 1).to(imgs.device)
    return (imgs - mean) / std

def _mean_cosine(pred_feats: torch.Tensor, target_feats: torch.Tensor) -> float:
    pred_feats = F.normalize(pred_feats.float(), dim=-1)
    target_feats = F.normalize(target_feats.float(), dim=-1)
    return float((pred_feats * target_feats).sum(-1).mean())

def _center_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=-1, keepdim=True)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

def correlation_distance(pred_feats: torch.Tensor, target_feats: torch.Tensor) -> float:
    r = (_center_normalize(pred_feats) * _center_normalize(target_feats)).sum(-1)
    return float((1.0 - r).mean())

def two_way_identification(pred_feats: torch.Tensor, target_feats: torch.Tensor) -> float:
    p = _center_normalize(pred_feats)
    t = _center_normalize(target_feats)
    sim = p @ t.T
    correct = sim.diag().unsqueeze(1)
    if sim.shape[0] < 2:
        return 1.0
    foil_mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    foil = sim[foil_mask].view(sim.shape[0], sim.shape[0] - 1)
    return float((correct > foil).float().mean())

@torch.no_grad()
def retrieval_metrics(pred_emb: torch.Tensor, target_emb: torch.Tensor,
                      batch_size: int = 300, seed: int = 0,
                      device=None) -> dict[str, float]:
    p = pred_emb.flatten(1)
    t = target_emb.flatten(1)
    n = p.shape[0]
    if n < 2:
        return {"retrieval_fwd": float("nan"), "retrieval_bwd": float("nan"),
                "retrieval_pool": 0, "retrieval_pools": 0}
    pool = min(batch_size, n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_pools = max(1, n // pool)
    fwd = bwd = 0.0
    for i in range(n_pools):
        idx = perm[i * pool:(i + 1) * pool]
        dev = device or p.device
        pb = F.normalize(p[idx].to(dev, torch.float32), dim=-1)
        tb = F.normalize(t[idx].to(dev, torch.float32), dim=-1)
        sim = tb @ pb.T
        labels = torch.arange(len(idx), device=sim.device)
        fwd += float((sim.argmax(dim=1) == labels).float().mean())
        bwd += float((sim.T.argmax(dim=1) == labels).float().mean())
    return {"retrieval_fwd": fwd / n_pools, "retrieval_bwd": bwd / n_pools,
            "retrieval_pool": pool, "retrieval_pools": n_pools}

def _get_clip(device):
    if "clip" not in _MODEL_CACHE:
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        m = m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE["clip"] = m
    return _MODEL_CACHE["clip"].to(device)

def _get_alexnet(device):
    if "alexnet" not in _MODEL_CACHE:
        from torchvision.models import alexnet, AlexNet_Weights
        m = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE["alexnet"] = m
    return _MODEL_CACHE["alexnet"].to(device)

def _get_inception(device):
    if "inception" not in _MODEL_CACHE:
        from torchvision.models import inception_v3, Inception_V3_Weights
        m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE["inception"] = m
    return _MODEL_CACHE["inception"].to(device)

def _get_effnet(device):
    if "effnet" not in _MODEL_CACHE:
        from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
        m = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE["effnet"] = m
    return _MODEL_CACHE["effnet"].to(device)

def _get_swav(device):
    if "swav" not in _MODEL_CACHE:
        try:
            import torch.hub
            m = torch.hub.load("facebookresearch/swav:main", "resnet50")
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            _MODEL_CACHE["swav"] = m
        except Exception as e:
            print(f"[metrics_full] WARNING: failed to load SwAV ({e}); skipping SwAV metrics.")
            _MODEL_CACHE["swav"] = None
    m = _MODEL_CACHE["swav"]
    return None if m is None else m.to(device)

@torch.no_grad()
def _clip_feature_pair(pred: torch.Tensor, target: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    clip_enc = _get_clip(device)
    pi = F.interpolate(pred.cpu(), 224, mode="bilinear", align_corners=False)
    gt = F.interpolate(target.cpu(), 224, mode="bilinear", align_corners=False)
    pi = _normalise(pi, _CLIP_MEAN, _CLIP_STD).to(device)
    gt = _normalise(gt, _CLIP_MEAN, _CLIP_STD).to(device)
    ep = clip_enc.encode_image(pi).float()
    et = clip_enc.encode_image(gt).float()
    return ep, et

@torch.no_grad()
def clip_similarity(pred: torch.Tensor, target: torch.Tensor, device) -> float:
    ep, et = _clip_feature_pair(pred, target, device)
    return _mean_cosine(ep, et)

@torch.no_grad()
def _alexnet_features(imgs: torch.Tensor, layer: int, device) -> torch.Tensor:
    m = _get_alexnet(device)
    x = _normalise(imgs.to(device), _IN_MEAN, _IN_STD)
    for i, blk in enumerate(m.features):
        x = blk(x)
        if i == layer:
            break
    return x.flatten(1)

@torch.no_grad()
def _alexnet_feature_pair(pred: torch.Tensor, target: torch.Tensor, layer: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    pi = F.interpolate(pred.cpu(), 224, mode="bilinear", align_corners=False)
    gt = F.interpolate(target.cpu(), 224, mode="bilinear", align_corners=False)
    fp = _alexnet_features(pi, layer, device)
    ft = _alexnet_features(gt, layer, device)
    return fp, ft

@torch.no_grad()
def alexnet_layer2_similarity(pred: torch.Tensor, target: torch.Tensor, device) -> float:
    fp, ft = _alexnet_feature_pair(pred, target, ALEXNET_EARLY_LAYER, device)
    return _mean_cosine(fp, ft)

@torch.no_grad()
def alexnet_layer5_similarity(pred: torch.Tensor, target: torch.Tensor, device) -> float:
    fp, ft = _alexnet_feature_pair(pred, target, ALEXNET_MID_LAYER, device)
    return _mean_cosine(fp, ft)

@torch.no_grad()
def _inception_feature_pair(pred: torch.Tensor, target: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    m = _get_inception(device)
    pi = F.interpolate(pred.cpu(), 299, mode="bilinear", align_corners=False)
    gt = F.interpolate(target.cpu(), 299, mode="bilinear", align_corners=False)
    pi = _normalise(pi, _IN_MEAN, _IN_STD).to(device)
    gt = _normalise(gt, _IN_MEAN, _IN_STD).to(device)

    def _feat(x):
        x = m.Conv2d_1a_3x3(x); x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x); x = m.maxpool1(x)
        x = m.Conv2d_3b_1x1(x); x = m.Conv2d_4a_3x3(x)
        x = m.maxpool2(x)
        x = m.Mixed_5b(x); x = m.Mixed_5c(x); x = m.Mixed_5d(x)
        x = m.Mixed_6a(x); x = m.Mixed_6b(x); x = m.Mixed_6c(x)
        x = m.Mixed_6d(x); x = m.Mixed_6e(x)
        x = m.Mixed_7a(x); x = m.Mixed_7b(x); x = m.Mixed_7c(x)
        x = m.avgpool(x)
        return x.flatten(1)

    fp = _feat(pi)
    ft = _feat(gt)
    return fp, ft

@torch.no_grad()
def inception_similarity(pred: torch.Tensor, target: torch.Tensor, device) -> float:
    fp, ft = _inception_feature_pair(pred, target, device)
    return _mean_cosine(fp, ft)

@torch.no_grad()
def _effnet_feature_pair(pred: torch.Tensor, target: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    m = _get_effnet(device)
    pi = F.interpolate(pred.cpu(), 240, mode="bilinear", align_corners=False)
    gt = F.interpolate(target.cpu(), 240, mode="bilinear", align_corners=False)
    pi = _normalise(pi, _IN_MEAN, _IN_STD).to(device)
    gt = _normalise(gt, _IN_MEAN, _IN_STD).to(device)

    def _feat(x):
        x = m.features(x)
        x = m.avgpool(x)
        return x.flatten(1)

    fp = _feat(pi)
    ft = _feat(gt)
    return fp, ft

@torch.no_grad()
def effnet_distance(pred: torch.Tensor, target: torch.Tensor, device) -> float:
    fp, ft = _effnet_feature_pair(pred, target, device)
    return correlation_distance(fp, ft)

@torch.no_grad()
def _swav_feature_pair(pred: torch.Tensor, target: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor] | None:
    m = _get_swav(device)
    if m is None:
        return None
    pi = F.interpolate(pred.cpu(), 224, mode="bilinear", align_corners=False)
    gt = F.interpolate(target.cpu(), 224, mode="bilinear", align_corners=False)
    pi = _normalise(pi, _IN_MEAN, _IN_STD).to(device)
    gt = _normalise(gt, _IN_MEAN, _IN_STD).to(device)

    def _feat(x):
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        x = m.avgpool(x)
        return x.flatten(1)

    fp = _feat(pi)
    ft = _feat(gt)
    return fp, ft

@torch.no_grad()
def swav_distance(pred: torch.Tensor, target: torch.Tensor, device) -> float:
    feats = _swav_feature_pair(pred, target, device)
    if feats is None:
        raise RuntimeError("SwAV is unavailable")
    fp, ft = feats
    return correlation_distance(fp, ft)

@torch.no_grad()
def evaluate_full(
    pred_images: torch.Tensor,
    target_images: torch.Tensor,
    device,
    skip: list[str] | None = None,
) -> dict[str, float]:
    skip = skip or []
    metrics: dict[str, float] = {}

    metrics["PixCorr"] = pixel_correlation(pred_images, target_images)
    metrics["SSIM"]    = ssim_grayscale(pred_images.cpu(), target_images.cpu())

    if "AlexNet(2)" not in skip:
        fp, ft = _alexnet_feature_pair(pred_images, target_images, ALEXNET_EARLY_LAYER, device)
        metrics["AlexNet(2)"] = _mean_cosine(fp, ft)
        metrics["AlexNet(2)_2way"] = two_way_identification(fp, ft)
    if "AlexNet(5)" not in skip:
        fp, ft = _alexnet_feature_pair(pred_images, target_images, ALEXNET_MID_LAYER, device)
        metrics["AlexNet(5)"] = _mean_cosine(fp, ft)
        metrics["AlexNet(5)_2way"] = two_way_identification(fp, ft)
    if "Inception" not in skip:
        fp, ft = _inception_feature_pair(pred_images, target_images, device)
        metrics["Inception"] = _mean_cosine(fp, ft)
        metrics["Inception_2way"] = two_way_identification(fp, ft)
    if "CLIP" not in skip:
        fp, ft = _clip_feature_pair(pred_images, target_images, device)
        metrics["CLIP"] = _mean_cosine(fp, ft)
        metrics["CLIP_2way"] = two_way_identification(fp, ft)
    if "EffNet-B" not in skip:
        fp, ft = _effnet_feature_pair(pred_images, target_images, device)
        metrics["EffNet-B"] = correlation_distance(fp, ft)
        metrics["EffNet-B_2way"] = two_way_identification(fp, ft)
    if "SwAV" not in skip:
        feats = _swav_feature_pair(pred_images, target_images, device)
        if feats is None:
            skip.append("SwAV")
        else:
            fp, ft = feats
            metrics["SwAV"] = correlation_distance(fp, ft)
            metrics["SwAV_2way"] = two_way_identification(fp, ft)

    return metrics

LOWER_IS_BETTER = ("EffNet-B", "SwAV")

NSD_REFERENCE = {
    "MindEye2 (1 subj, 40h)": {
        "PixCorr": 0.322, "SSIM": 0.431, "AlexNet(2)_2way": 0.964,
        "AlexNet(5)_2way": 0.984, "Inception_2way": 0.942, "CLIP_2way": 0.930,
        "EffNet-B": 0.619, "SwAV": 0.344,
    },
    "Ozcelik & VanRullen 2023": {
        "PixCorr": 0.273, "SSIM": 0.365, "AlexNet(2)_2way": 0.947,
        "AlexNet(5)_2way": 0.978, "Inception_2way": 0.936, "CLIP_2way": 0.909,
        "EffNet-B": 0.728, "SwAV": 0.421,
    },
    "Takagi & Nishimoto 2023": {
        "PixCorr": 0.246, "SSIM": 0.410, "AlexNet(2)_2way": 0.789,
        "AlexNet(5)_2way": 0.869, "Inception_2way": 0.867, "CLIP_2way": 0.821,
        "EffNet-B": 0.811, "SwAV": 0.504,
    },
}

def format_comparison(metrics: dict[str, float], reference: dict | None = None) -> str:
    reference = NSD_REFERENCE if reference is None else reference
    cols = ["PixCorr", "SSIM", "AlexNet(2)_2way", "AlexNet(5)_2way",
            "Inception_2way", "CLIP_2way", "EffNet-B", "SwAV"]
    head = ["method"] + [c.replace("_2way", "") for c in cols]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    def row(name, d):
        cells = [f"{d[c]:.3f}" if c in d and d[c] == d[c] else "--" for c in cols]
        return "| " + " | ".join([name] + cells) + " |"
    lines.append(row("**ours**", metrics))
    for name, d in reference.items():
        lines.append(row(name, d))
    lines.append("")
    lines.append("EffNet-B and SwAV are correlation distances: LOWER is better.")
    return "\n".join(lines)
