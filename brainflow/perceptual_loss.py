from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class LPIPS(nn.Module):
    def __init__(self, net='alex'):
        super().__init__()
        try:
            import lpips
            self.lpips = lpips.LPIPS(net=net, verbose=False)

            for p in self.lpips.parameters():
                p.requires_grad = False
        except ImportError:
            raise ImportError(
                "lpips package not found. Install with: pip install lpips"
            )

    def forward(self, pred, target):
        return self.lpips(pred, target).mean()

class L1PixelLoss(nn.Module):

    def forward(self, pred, target):
        return F.l1_loss(pred, target)

def build_perceptual_loss(loss_type: str) -> nn.Module | None:
    if loss_type == "none":
        return None
    elif loss_type == "lpips":
        return LPIPS(net='alex')
    elif loss_type == "l1":
        return L1PixelLoss()
    else:
        raise ValueError(f"Unknown perceptual loss type: {loss_type!r}")
