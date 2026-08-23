"""MindEye2's low-level pathway, imported rather than reimplemented.

Seven attempts at our own low-level head have failed, none ever beating
blur_mse 1.0 (the score for predicting the channel mean). MindEye2's version
reaches PixCorr 0.322 on the same subjects, the same voxels and the same 982
test images, so the information is there and a ~10M-parameter module can get
it. This runs their module instead of writing an eighth of ours.

Three things they do that none of our seven attempts did, all visible in
third_party/MindEyeV2/src/models.py:15-46,100-118 and their Train.ipynb:

* **The target is the latent of the REAL image**, not a blurred one:
  ``autoenc.encode(2*img-1).latent_dist.mode() * 0.18215`` at 4x28x28. The
  blurriness comes from a 7x7 spatial bottleneck in the head, not from
  softening the target. We blurred the target and then gave the head 69M
  parameters to fit it -- a target with less across-image variance makes
  predicting the mean relatively MORE attractive, which is what we measured.
* **An auxiliary contrastive term on intermediate feature maps**:
  ``b_maps_projector`` emits (B, 49, 512) and is scored by ``soft_cont_loss``
  against ConvNeXt-XL features of the image and an augmented copy. Nothing in
  our attempts made per-sample distinctness necessary.
* **Joint training with the trunk.** Their trunk serves the blurry head and
  the CLIP head at once. Ours was trained for CLIP and frozen, which is why
  feeding it to the low head (--ll-use-backbone) made things WORSE, 1.7055
  against raw voxels' 1.1270: by then the trunk had discarded what the head
  needed.

The class definitions are imported from their tree, not copied, so there is no
transcription surface. Only ``RidgeRegression`` is written here, because it
lives inline in their notebook rather than in models.py, and it is one Linear.

Their bar, from Train.ipynb: ``blurry_pixcorr`` must exceed **0.456** to beat
MindEye v1's low-level subject-1 result. That is the pass mark, not our 1.0.
"""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Their normalisation, reproduced verbatim -- note std[0] is 0.228 where
# ImageNet's is 0.229. The ConvNeXt was used with these values, so matching
# them matters more than being right about ImageNet.
CNX_MEAN = (0.485, 0.456, 0.406)
CNX_STD = (0.228, 0.224, 0.225)
SD_LATENT_SCALE = 0.18215
BLURRY_PIXCORR_BAR = 0.456


def add_mindeye_to_path(mindeye_src) -> pathlib.Path:
    src = pathlib.Path(mindeye_src).resolve()
    if not (src / "models.py").is_file():
        raise SystemExit(f"no models.py under {src}; pass --mindeye-src")
    for p in (str(src), str(src / "generative_models")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return src


def _import_brainnetwork(max_stubs: int = 16):
    """Import their BrainNetwork without installing what only their other classes need.

    models.py opens with `import clip` (OpenAI CLIP), for the Clipper class, and
    pulls in more for BrainDiffusionPrior. BrainNetwork itself needs only torch,
    nn and `from diffusers.models.vae import Decoder`, all of which we have. So
    stub whatever is missing, one ImportError at a time, rather than adding
    dependencies we never call.

    The stub answers any attribute with a dummy class, so `from X import Y` also
    resolves. That is only safe because we verify afterwards that the one piece
    BrainNetwork genuinely depends on -- the diffusers Decoder -- is the real
    class and not something a stub handed back.
    """
    import importlib
    import types

    class _Stub(types.ModuleType):
        # __path__ must be a real iterable: the import machinery walks it when
        # resolving `from pkg.sub import X`, and a stub that answers every
        # attribute with a class makes it raise TypeError instead of ImportError,
        # escaping the retry loop entirely. Dunders must fall through for the
        # same reason.
        __path__: list = []

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return type(name, (), {})

    stubbed = []
    for _ in range(max_stubs):
        try:
            mod = importlib.import_module("models")
            break
        except ImportError as e:
            name = getattr(e, "name", None) or str(e).split("'")[1]
            if name in (None, "models") or name in stubbed:
                raise
            sys.modules[name] = _Stub(name)
            stubbed.append(name)
    else:
        raise ImportError(f"gave up after stubbing {stubbed}")

    from diffusers.models.vae import Decoder as RealDecoder
    if getattr(mod, "Decoder", None) is not RealDecoder:
        raise ImportError(
            "their models.py did not get the real diffusers Decoder -- a stub "
            f"shadowed it. Stubbed: {stubbed}. The blurry head's upsampler IS "
            "that Decoder, so this would train a different architecture.")
    if stubbed:
        print(f"✓ stubbed unused MindEyeV2 imports: {', '.join(stubbed)}", flush=True)
    return mod.BrainNetwork


def soft_cont_loss(student_preds, teacher_preds, teacher_aug_preds, temp=0.125):
    """Verbatim from third_party/MindEyeV2/src/utils.py:309-319.

    The soft targets come from teacher-vs-augmented-teacher, so the student is
    asked to reproduce the teacher's own similarity structure rather than a
    hard identity -- which is what keeps it from collapsing onto one answer.
    """
    tt = (teacher_preds @ teacher_aug_preds.T) / temp
    tt_t = (teacher_aug_preds @ teacher_preds.T) / temp
    st = (student_preds @ teacher_aug_preds.T) / temp
    st_t = (teacher_aug_preds @ student_preds.T) / temp
    loss1 = -(st.log_softmax(-1) * tt.softmax(-1)).sum(-1).mean()
    loss2 = -(st_t.log_softmax(-1) * tt_t.softmax(-1)).sum(-1).mean()
    return (loss1 + loss2) / 2


class MindEyeBlurry(nn.Module):
    """Per-subject ridge into their BrainNetwork, blurry head only.

    ``seq_len=1``: their sequence dimension stacks multiple timepoints, which we
    do not have -- one trial is one vector. ``blin1`` maps ``h*seq_len -> 3136``
    and 3136/49 = 64, so the 64x7x7 bottleneck is preserved at any seq_len.

    ``clip_scale`` must stay > 0: their forward returns ``backbone, c, b`` and
    ``c`` is only assigned inside ``if self.clip_scale > 0``, so setting it to
    zero raises NameError. We take the blurry output and drop the rest.
    """

    def __init__(self, voxels: dict[int, int], mindeye_src, h: int = 4096,
                 n_blocks: int = 4, drop: float = 0.15, seq_len: int = 1):
        super().__init__()
        add_mindeye_to_path(mindeye_src)
        BrainNetwork = _import_brainnetwork()

        self.h, self.seq_len = h, seq_len
        self.ridge = nn.ModuleDict({str(s): nn.Linear(v, h * seq_len)
                                    for s, v in voxels.items()})
        self.net = BrainNetwork(h=h, in_dim=max(voxels.values()), out_dim=768,
                                seq_len=seq_len, n_blocks=n_blocks, drop=drop,
                                clip_size=768, blurry_recon=True, clip_scale=1)

    def forward(self, x: torch.Tensor, subject: int):
        r = self.ridge[str(subject)](x).view(x.shape[0], self.seq_len, self.h)
        _, _, b = self.net(r)
        latent, aux = b                      # (B,4,28,28), (B,49,512)
        return latent, aux


def load_autoenc(ckpt_path, device):
    """Their AutoencoderKL config, verbatim from Train.ipynb."""
    from diffusers import AutoencoderKL
    ae = AutoencoderKL(
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ae.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    ae.eval().requires_grad_(False)
    return ae.to(device)


def load_convnext(ckpt_path, mindeye_src, device):
    add_mindeye_to_path(mindeye_src)
    from autoencoder.convnext import ConvnextXL  # noqa: E402
    cnx = ConvnextXL(str(ckpt_path))
    cnx.eval().requires_grad_(False)
    return cnx.to(device)


def build_blur_augs():
    """Their kornia stack. Returns None if kornia is unavailable, in which case
    the augmented view falls back to the image itself and the contrastive term
    degenerates -- so the caller warns rather than silently training a
    different objective."""
    try:
        import kornia
        from kornia.augmentation.container import AugmentationSequential
    except Exception:
        return None
    return AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.1),
        kornia.augmentation.RandomSolarize(p=0.1),
        kornia.augmentation.RandomResizedCrop((224, 224), scale=(0.9, 0.9),
                                              ratio=(1, 1), p=1.0),
        data_keys=["input"],
    )


@torch.no_grad()
def encode_target(images01: torch.Tensor, autoenc) -> torch.Tensor:
    """images01 in [0,1] -> the latent their head regresses onto."""
    return autoenc.encode(2 * images01 - 1).latent_dist.mode() * SD_LATENT_SCALE


def _normalise(images01, device):
    m = torch.tensor(CNX_MEAN, device=device).view(1, 3, 1, 1)
    s = torch.tensor(CNX_STD, device=device).view(1, 3, 1, 1)
    return (images01 - m) / s


def blurry_loss(pred_latent, aux, images01, target_latent, cnx, augs,
                cont_weight: float = 0.1):
    """L1 to the latent plus their feature-level contrastive term."""
    out = {"l1": F.l1_loss(pred_latent, target_latent)}
    if cnx is None:
        out["loss"] = out["l1"]
        return out
    with torch.no_grad():
        img_norm = _normalise(images01, images01.device)
        aug = augs(images01) if augs is not None else images01
        img_aug = _normalise(aug, images01.device)
        _, cnx_embeds = cnx(img_norm)
        _, cnx_aug = cnx(img_aug)
    d = cnx_embeds.shape[-1]
    out["cont"] = soft_cont_loss(
        F.normalize(aux.reshape(-1, aux.shape[-1]).float(), dim=-1),
        F.normalize(cnx_embeds.reshape(-1, d).float(), dim=-1),
        F.normalize(cnx_aug.reshape(-1, d).float(), dim=-1),
        temp=0.2)
    out["loss"] = out["l1"] + cont_weight * out["cont"]
    return out


@torch.no_grad()
def blurry_pixcorr(pred_latent, images01, autoenc) -> float:
    """Their eval: decode the predicted latent and correlate with the image.

    Pass mark 0.456 (Train.ipynb: "needs >.456 to beat low-level subj01
    results in mindeye v1").
    """
    rec = (autoenc.decode(pred_latent / SD_LATENT_SCALE).sample / 2 + 0.5).clamp(0, 1)
    if rec.shape[-2:] != images01.shape[-2:]:
        rec = F.interpolate(rec, images01.shape[-2:], mode="bilinear",
                            align_corners=False)
    a = rec.flatten(1).float()
    b = images01.flatten(1).float()
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)
            ).mean().item()


@torch.no_grad()
def decode_blurry(pred_latent, autoenc, out_size=None) -> torch.Tensor:
    """Predicted latent -> blurry RGB in [0,1], for decoder_sgm's init_image.

    The handoff is deliberately in PIXEL space: their SD-VAE works at 4x28x28
    and our SDXL init path at 4x96x96, and nothing has to reconcile the two if
    the module simply hands over an image.
    """
    rec = (autoenc.decode(pred_latent / SD_LATENT_SCALE).sample / 2 + 0.5).clamp(0, 1)
    if out_size is not None and rec.shape[-1] != out_size:
        rec = F.interpolate(rec, out_size, mode="bilinear", align_corners=False)
    return rec
