"""Frozen DINOv2 embedding extraction for every saved run.

For each (dataset, seed) we extract DINOv2 ViT-S/14 CLS embeddings for the
*whole* CIFAR train split (50k images), keyed by ``sample_id``.  Per-run
subsets are then gathered by ``sample_id`` (each seed uses a different
train/val permutation, so the runs do not share an index set).

Two properties matter for the revision:

* the backbone is **frozen** and **label-free** (self-supervised pretraining,
  no fine-tuning, no labels passed anywhere),
* embeddings are cached on disk so the whole revision is re-runnable and cheap.

We also cache the *native* ResNet-18 penultimate embedding from each run's saved
checkpoint (``model_*.pt``) under the same key, so the revision can compare
DINO-Q against both native and prototype quality on identical runs.

Output layout::

    <out>/dino/<dataset>/seed<seed>.npy   (n_train_dataset, 384) float16
    <out>/dino/<dataset>/seed<seed>.ids.npy  (n_train_dataset,) int32
    <out>/native/<dataset>/<noise>/seed<seed>.npy  (n_run,) float32
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from revq.dinov2 import DINO_MEAN, DINO_STD, DEFAULT_CKPT, load_dinov2  # noqa: E402

CIFAR_ROOT = Path("/root/.cache/quality_noise")
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "cifar10n": 10, "cifar100n": 100}
# Human-annotation datasets reuse the base CIFAR image pool.
IMAGE_DATASET = {"cifar10": "cifar10", "cifar100": "cifar100",
                 "cifar10n": "cifar10", "cifar100n": "cifar100"}


def _train_images(dataset: str) -> np.ndarray:
    """Return the CIFAR train images as uint8 (N, 3, 32, 32)."""
    name = IMAGE_DATASET[dataset]
    root = CIFAR_ROOT / name
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    ds = cls(root=str(root), train=True, download=False)
    imgs = np.asarray(ds.data)                    # (N, 32, 32, 3)
    return np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))


class _TensorDS(torch.utils.data.Dataset):
    def __init__(self, imgs_chw: np.ndarray, size: int) -> None:
        self.imgs = imgs_chw
        self.size = size
        self.mean = torch.tensor(DINO_MEAN).view(3, 1, 1)
        self.std = torch.tensor(DINO_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> torch.Tensor:
        x = torch.from_numpy(self.imgs[i]).float().div_(255.0)
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(self.size, self.size), mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        return (x - self.mean) / self.std


@torch.no_grad()
def extract_dino(imgs: np.ndarray, batch: int = 256, size: int = 224,
                 device: str = "cuda", workers: int = 8, ckpt: Path = DEFAULT_CKPT) -> np.ndarray:
    """CLS embeddings (N, 384) float16 for the given uint8 CHW images."""
    model = load_dinov2(ckpt, device=device)
    ds = _TensorDS(imgs, size)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )
    out = []
    t0 = time.time()
    for bi, x in enumerate(dl):
        x = x.to(device, non_blocking=True)
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                feats = model(x)
        else:
            feats = model(x)
        out.append(feats[:, 0].float().cpu().numpy().astype(np.float16))
        if bi % 40 == 0:
            done = min((bi + 1) * batch, len(imgs))
            print(f"    dino {done}/{len(imgs)} ({time.time()-t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


class _Penultimate(torch.nn.Module):
    """torchvision ResNet with ``fc`` stripped, returning the pooled 512-d feature.

    The project's saved checkpoints are plain ``torchvision.models.resnet18``
    state dicts with a replaced ``fc``, so we reuse the same architecture and
    simply read the input to ``fc`` (i.e. after ``avgpool``).
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        import torchvision
        net = torchvision.models.resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
        self.net = net
        self.feat_dim = net.fc.in_features

    def load(self, sd: dict) -> None:
        self.net.load_state_dict(sd, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.net
        x = n.conv1(x); x = n.bn1(x); x = n.relu(x); x = n.maxpool(x)
        x = n.layer1(x); x = n.layer2(x); x = n.layer3(x); x = n.layer4(x)
        x = n.avgpool(x)
        return torch.flatten(x, 1)


def extract_native(dataset: str, noise: str, seed: int, results_root: Path,
                   device: str = "cuda", batch: int = 512) -> np.ndarray | None:
    """Penultimate ResNet-18 features for one run, aligned to the run sample order.

    Uses the saved ``model_<hash>.pt`` checkpoint.  Sample order is taken from
    the run's own saved ``embeddings_<hash>.npy`` length and the sorted
    ``sample_id`` column of its quality parquet (the same convention
    ``io_utils.load_run`` uses).  Returns None when the ordering cannot be
    established.
    """
    run_dir = results_root / "traces" / dataset / noise / f"seed{seed}"
    ckpts = sorted(run_dir.glob("model_*.pt"))
    emb_files = sorted(run_dir.glob("embeddings_*.npy"))
    if not ckpts or not emb_files:
        return None
    want_n = int(np.load(emb_files[0], mmap_mode="r").shape[0])

    import pandas as pd
    qfiles = sorted((results_root / "quality" / dataset / noise / f"seed{seed}").glob("quality_*.parquet"))
    if not qfiles:
        return None
    sids = np.sort(pd.read_parquet(qfiles[0], columns=["sample_id"])["sample_id"].to_numpy().astype(int))
    if len(sids) != want_n:
        print(f"    [native] sample_id/embeddings length mismatch ({len(sids)} vs {want_n})", flush=True)
        return None

    imgs = _train_images(dataset)[sids]
    ds = _TensorDS(imgs, 32)  # native resolution; ImageNet normalisation
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True)

    model = _Penultimate(NUM_CLASSES[dataset])
    sd = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load(sd)
    model = model.to(device).eval()

    feats = []
    with torch.no_grad():
        for x in dl:
            feats.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/root/quality_noise/results/revision/embeddings"))
    ap.add_argument("--results-root", type=Path, default=Path("/root/quality_noise/results"))
    ap.add_argument("--datasets", type=str, default="cifar10,cifar100,cifar10n")
    ap.add_argument("--seeds", type=str, default="all")
    ap.add_argument("--what", type=str, default="dino", choices=["dino", "native", "both"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dino-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip seeds whose embedding file already exists")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    want_seeds = None if args.seeds == "all" else {int(s) for s in args.seeds.split(",")}

    if args.what in ("dino", "both"):
        for ds in datasets:
            out_dir = args.out / "dino" / ds
            out_dir.mkdir(parents=True, exist_ok=True)
            # CIFAR-10N reuses CIFAR-10 images: identical embeddings.
            seeds = sorted({int(p.name[4:]) for base in (args.results_root / "traces" / ds).glob("*")
                            for p in base.glob("seed*") if p.is_dir()})
            if want_seeds is not None:
                seeds = [s for s in seeds if s in want_seeds]
            if args.skip_existing:
                seeds = [s for s in seeds if not (out_dir / f"seed{s}.npy").exists()]
                if not seeds:
                    print(f"[{ds}] all DINO embeddings already present", flush=True)
                    continue
            if not seeds:
                print(f"[{ds}] no runs found, skipping DINO", flush=True)
                continue
            imgs = _train_images(ds)
            print(f"[{ds}] extracting DINOv2 for {len(imgs)} images (needed seeds {seeds})", flush=True)
            emb = extract_dino(imgs, batch=args.batch, size=args.dino_size, device=args.device)
            for s in seeds:
                np.save(out_dir / f"seed{s}.npy", emb)
            np.save(out_dir / "sample_ids.npy", np.arange(len(imgs), dtype=np.int32))
            print(f"[{ds}] wrote {len(seeds)} seed files, shape {emb.shape}", flush=True)

    if args.what in ("native", "both"):
        for ds in datasets:
            for noise_dir in sorted((args.results_root / "traces" / ds).glob("*")):
                noise = noise_dir.name
                for p in sorted(noise_dir.glob("seed*")):
                    if not p.is_dir():
                        continue
                    seed = int(p.name[4:])
                    if want_seeds is not None and seed not in want_seeds:
                        continue
                    out_path = args.out / "native" / ds / noise / f"seed{seed}.npy"
                    if out_path.exists():
                        print(f"[native] {ds}/{noise}/seed{seed} cached", flush=True)
                        continue
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    t0 = time.time()
                    f = extract_native(ds, noise, seed, args.results_root, device=args.device)
                    if f is None:
                        print(f"[native] {ds}/{noise}/seed{seed} SKIP (no ckpt/order)", flush=True)
                        continue
                    np.save(out_path, f.astype(np.float32))
                    print(f"[native] {ds}/{noise}/seed{seed} {f.shape} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
