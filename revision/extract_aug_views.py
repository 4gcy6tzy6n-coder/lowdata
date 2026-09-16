"""Augmentation embeddings for a second label-free quality family.

`compute_independent_q` builds Q from *local density* in a frozen embedding.  A
reviewer will ask whether the near-zero gap is specific to that choice, so we
need a second, structurally different label-free quality:

    Q^aug_i = cos( h_i , h_i^(m) )   averaged over M augmented views

i.e. representation stability under a label-preserving perturbation.  Like
density, it never reads a label; unlike density it measures *stability* rather
than typicality.

We store one clean DINOv2 embedding per image (already on disk) plus embeddings
of M augmented views, then compute the mean cosine similarity on demand.  Views
are deterministic (fixed RNG seed) so the result is reproducible.

Output: ``<emb>/dino/<dataset>/aug<idx>.npy``  (50000, 384) float16 each.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from revq.dinov2 import DINO_MEAN, DINO_STD, DEFAULT_CKPT, load_dinov2  # noqa: E402
from extract_embeddings import _train_images  # noqa: E402

EMB = Path("/root/quality_noise/results/revision/embeddings/dino")
# CIFAR-10N and CIFAR-100N reuse the CIFAR-10 / CIFAR-100 image pools.
IMAGE_POOL = {"cifar10": "cifar10", "cifar100": "cifar100",
              "cifar10n": "cifar10", "cifar100n": "cifar100"}


class _AugView(torch.utils.data.Dataset):
    """Deterministic augmented view ``idx`` of each image."""

    def __init__(self, imgs_chw: np.ndarray, size: int, view_idx: int,
                 seed: int = 1234) -> None:
        self.imgs = imgs_chw
        self.size = size
        self.view_idx = view_idx
        self.seed = seed
        self.mean = torch.tensor(DINO_MEAN).view(3, 1, 1)
        self.std = torch.tensor(DINO_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> torch.Tensor:
        # Per-(image, view) deterministic RNG: reproducible across runs and
        # independent of DataLoader worker scheduling.
        rng = np.random.default_rng((self.seed, self.view_idx, int(i)))
        x = torch.from_numpy(self.imgs[i]).float().div_(255.0)
        # Random crop with 4px reflect padding, then optional flip.
        pad = 4
        x = torch.nn.functional.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
        top = int(rng.integers(0, 2 * pad + 1))
        left = int(rng.integers(0, 2 * pad + 1))
        x = x[:, top:top + 32, left:left + 32]
        if rng.random() < 0.5:
            x = torch.flip(x, dims=[2])
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(self.size, self.size), mode="bicubic",
            align_corners=False).squeeze(0)
        return (x - self.mean) / self.std


@torch.no_grad()
def extract_view(imgs: np.ndarray, view_idx: int, batch: int = 256, size: int = 224,
                 device: str = "cuda", workers: int = 8) -> np.ndarray:
    model = load_dinov2(DEFAULT_CKPT, device=device)
    ds = _AugView(imgs, size, view_idx)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=workers, pin_memory=True,
                                     persistent_workers=workers > 0)
    out = []
    t0 = time.time()
    for bi, x in enumerate(dl):
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            feats = model(x)
        out.append(feats[:, 0].float().cpu().numpy().astype(np.float16))
        if bi % 60 == 0:
            print(f"    view{view_idx} {min((bi+1)*batch, len(imgs))}/{len(imgs)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=3, help="number of augmented views M")
    ap.add_argument("--datasets", type=str, default="cifar10,cifar100")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        pool = IMAGE_POOL[ds]
        out_dir = EMB / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        need = [v for v in range(args.views) if not (out_dir / f"aug{v}.npy").exists()]
        if not need:
            print(f"[{ds}] all {args.views} augmented views already present", flush=True)
            continue
        imgs = _train_images(pool)
        print(f"[{ds}] pool={pool} n={len(imgs)} extracting views {need}", flush=True)
        for v in need:
            t0 = time.time()
            emb = extract_view(imgs, v, batch=args.batch, device=args.device)
            np.save(out_dir / f"aug{v}.npy", emb)
            print(f"[{ds}] view {v} -> {emb.shape} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
