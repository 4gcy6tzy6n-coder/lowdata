"""Frozen DINOv2 ViT-S/14, implemented directly against the public checkpoint.

We deliberately avoid ``timm``/``xformers`` so the revision pipeline has no new
heavy dependencies: the released ``dinov2_vits14_pretrain.pth`` state dict maps
1:1 onto a plain ViT with LayerScale, which is ~120 lines of PyTorch.

The backbone is *frozen* and *label-free*: DINOv2 was pretrained with self-
supervision (no class labels), and we never fine-tune it.  This is the property
the revision needs: Q built from these embeddings satisfies Q not<- y_tilde and
(up to the pretraining corpus) Q not<- noisy-label training.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CKPT = Path("/root/quality_noise/models_cache/dinov2_vits14_pretrain.pth")

# ViT-S/14.  The released checkpoint stores positional embeddings for the
# pretraining resolution 518 (= 37*14), i.e. 1369 patch tokens; we build the
# module at that resolution and bicubically interpolate per input size.
_EMBED_DIM = 384
_DEPTH = 12
_NUM_HEADS = 6
_PATCH = 14
_IMG = 518


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = _IMG, patch_size: int = _PATCH, in_chans: int = 3,
                 embed_dim: int = _EMBED_DIM) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                       # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)       # (B, N, C)
        return x


def _interpolate_pos_encoding(x: torch.Tensor, pos_embed: torch.Tensor,
                              patch_size: int, grid: int) -> torch.Tensor:
    """Bicubic interpolation of the positional embedding to a new grid.

    Lets one checkpoint serve 224x224 (grid 16) and other resolutions.
    """
    npatch = x.shape[1] - 1
    if npatch == pos_embed.shape[1] - 1:
        return pos_embed
    dim = pos_embed.shape[-1]
    cls_pos = pos_embed[:, :1]
    patch_pos = pos_embed[:, 1:]
    w0 = h0 = int(math.sqrt(patch_pos.shape[1]))
    patch_pos = patch_pos.reshape(1, w0, h0, dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(grid, grid), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid * grid, dim)
    return torch.cat([cls_pos, patch_pos], dim=1)


class DinoV2ViT(nn.Module):
    """Minimal DINOv2 ViT matching the released checkpoint naming."""

    def __init__(self, img_size: int = _IMG, patch_size: int = _PATCH,
                 embed_dim: int = _EMBED_DIM, depth: int = _DEPTH,
                 num_heads: int = _NUM_HEADS) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        grid = x.shape[-1] // self.patch_size
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + _interpolate_pos_encoding(x, self.pos_embed, self.patch_size, grid)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # (B, 1+N, C)


def load_dinov2(ckpt: Path | str = DEFAULT_CKPT, device: str = "cuda") -> DinoV2ViT:
    """Load the frozen DINOv2 ViT-S/14 backbone in eval mode."""
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"DINOv2 checkpoint not found: {ckpt}")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = DinoV2ViT()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The checkpoint carries no buffers we care about; any *parameter* mismatch is fatal.
    bad_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if bad_missing:
        raise RuntimeError(f"DINOv2 load: missing params {bad_missing}")
    if unexpected:
        raise RuntimeError(f"DINOv2 load: unexpected keys {unexpected}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


# DINOv2 normalisation (ImageNet statistics).
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
