"""Perceptual loss implementations for BrainFlow."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS) loss.
    
    Wrapper around the 'lpips' package for perceptual loss.
    """
    def __init__(self, net='alex'):
        super().__init__()
        try:
            import lpips
            self.lpips = lpips.LPIPS(net=net, verbose=False)
            # Freeze LPIPS network
            for p in self.lpips.parameters():
                p.requires_grad = False
        except ImportError:
            raise ImportError(
                "lpips package not found. Install with: pip install lpips"
            )
    
    def forward(self, pred, target):
        """Compute LPIPS loss.
        
        Args:
            pred: (B, 3, H, W) predicted images in [-1, 1]
            target: (B, 3, H, W) target images in [-1, 1]
        
        Returns:
            Scalar loss value (mean over batch)
        """
        return self.lpips(pred, target).mean()


class L1PixelLoss(nn.Module):
    """Simple L1 pixel loss for color accuracy."""
    
    def forward(self, pred, target):
        """Compute L1 pixel loss.
        
        Args:
            pred: (B, 3, H, W) predicted images
            target: (B, 3, H, W) target images
        
        Returns:
            Scalar loss value
        """
        return F.l1_loss(pred, target)


def build_perceptual_loss(loss_type: str) -> nn.Module | None:
    """Build perceptual loss module.
    
    Args:
        loss_type: "none" | "lpips" | "l1"
    
    Returns:
        Loss module or None if loss_type == "none"
    """
    if loss_type == "none":
        return None
    elif loss_type == "lpips":
        return LPIPS(net='alex')
    elif loss_type == "l1":
        return L1PixelLoss()
    else:
        raise ValueError(f"Unknown perceptual loss type: {loss_type!r}")
