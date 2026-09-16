"""Dataset views that enforce the iron rule: clean labels and the noise mask
are evaluation-only.

``TrainingView`` physically does not hold ``clean_labels`` or ``mask`` — they
are not attributes and no code path can read them from a training view.
``EvaluationView`` additionally exposes ``clean_labels`` and ``mask`` and is
intended solely for evaluation / metrics computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from . import noise as noise_mod


# CIFAR-10N human annotation subsets. Each key maps to a human-annotated noise
# label array in CIFAR-10_human_anno.npz (real annotation noise, not synthetic).
CIFAR10N_SUBSETS: dict[str, str] = {
    "aggregate": "aggre_label",
    "worse": "worse_label",
    "random1": "random_label1",
    "random2": "random_label2",
    "random3": "random_label3",
}

# CIFAR-100N human annotation (Wei et al., 2022). Same construction as CIFAR-10N
# but with fine-grained 100-class labels: each image gets a single human label
# (noisy_label) and an expert-verified clean label (clean_label).
CIFAR100N_SUBSETS: dict[str, str] = {
    "human": "noisy_label",
}


def _resolve_data_root(cfg_dataset: dict, name: str) -> str:
    """Dataset root resolution: config root > env QUALITY_NOISE_DATA_ROOT > ~/.cache."""
    import os

    root = cfg_dataset.get("root")
    if root is None:
        env_root = os.environ.get("QUALITY_NOISE_DATA_ROOT")
        root = str(Path(env_root) / name) if env_root else str(Path.home() / ".cache" / "quality_noise" / name)
    return root


def load_raw_dataset(cfg_dataset: dict) -> tuple[Any, Any]:
    """Load raw CIFAR train/test sets (lazily downloads on first call).

    Dataset root resolution order: config ``root`` > env QUALITY_NOISE_DATA_ROOT
    (a directory containing per-dataset subdirs) > ~/.cache/quality_noise/<name>.
    Download can be disabled via env QUALITY_NOISE_DOWNLOAD=0 or config
    ``download: false`` (useful when the tarballs are pre-placed on a server).
    ``cifar10n`` reuses CIFAR-10 images (its test set drives evaluation).
    """
    import os

    import torchvision

    name = cfg_dataset["name"]
    if name in ("cifar10n", "cifar100n"):
        # CIFAR-10N / CIFAR-100N reuse the corresponding CIFAR images and their
        # held-out test sets; only the *training* labels are human annotations.
        base = dict(cfg_dataset)
        base["name"] = "cifar10" if name == "cifar10n" else "cifar100"
        base["root"] = None
        return load_raw_dataset(base)
    if name not in ("cifar10", "cifar100"):
        raise ValueError(f"Unsupported dataset: {name}")
    root = _resolve_data_root(cfg_dataset, name)
    download = bool(cfg_dataset.get("download", True))
    if os.environ.get("QUALITY_NOISE_DOWNLOAD", "") in ("0", "false", "False", "no"):
        download = False
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    train = cls(root=root, train=True, download=download)
    test = cls(root=root, train=False, download=download)
    return train, test


def load_cifar100n_labels(cfg_dataset: dict) -> dict[str, np.ndarray]:
    """Load CIFAR-100N human annotation labels (``CIFAR-100_human.pt``).

    Returns ``{"human": y_noisy, "clean": y_clean}`` as 0-99 class indices.
    """
    import os

    root = Path(_resolve_data_root(cfg_dataset, "cifar100n"))
    pt_path = root / "CIFAR-100_human.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"CIFAR-100N annotation file not found at {pt_path}. Download "
            "CIFAR-100_human.pt from UCSC-REAL/cifar-10-100n (data/)."
        )
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    def _to_index(arr):
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr.argmax(axis=1)
        return arr.reshape(-1).astype(int)

    out: dict[str, np.ndarray] = {}
    for key, pt_key in CIFAR100N_SUBSETS.items():
        if pt_key in data:
            out[key] = _to_index(data[pt_key])
    if "clean_label" in data:
        out["clean"] = _to_index(data["clean_label"])
    return out


def load_cifar10n_labels(cfg_dataset: dict) -> dict[str, np.ndarray]:
    """Load CIFAR-10 human annotation noise labels.

    Prefers ``CIFAR-10_human.pt`` (torch dict: clean_label, aggre_label,
    worse_label, random_label1/2/3 as (50000,) class indices); falls back to
    ``CIFAR-10_human_anno.npz`` (same keys). Each label is reduced to class
    indices if stored as one-hot. Returns ``{"subset_name": y_noisy, "clean": y_clean}``.
    """
    import os

    root = Path(_resolve_data_root(cfg_dataset, "cifar10n"))
    pt_path = root / "CIFAR-10_human.pt"
    npz_path = root / "CIFAR-10_human_anno.npz"
    if pt_path.exists():
        import torch

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    elif npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
    else:
        raise FileNotFoundError(
            f"CIFAR-10N annotation file not found in {root}. "
            "Place either CIFAR-10_human.pt or CIFAR-10_human_anno.npz there."
        )

    def _to_index(arr, key):
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] > 1:  # one-hot -> class index
            arr = arr.argmax(axis=1)
        if key in ("clean", "clean_label") and arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr.astype(int)

    out: dict[str, np.ndarray] = {}
    for key, npz_key in CIFAR10N_SUBSETS.items():
        if npz_key in data:
            out[key] = _to_index(data[npz_key], npz_key)
    if "clean_label" in data:
        out["clean"] = _to_index(data["clean_label"], "clean")
    return out


def to_numpy_arrays(raw: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert a torchvision CIFAR object to (images uint8 (N,3,32,32), labels)."""
    if hasattr(raw, "data") and hasattr(raw, "targets"):
        imgs = np.asarray(raw.data)
        labels = np.asarray(raw.targets)
        # torchvision returns HWC; transpose to CHW.
        if imgs.ndim == 4 and imgs.shape[3] == 3:
            imgs = imgs.transpose(0, 3, 1, 2)
        return imgs, labels
    raise TypeError(f"Unsupported raw dataset object: {type(raw)}")


class TrainingView(Dataset):
    """Holds only (x, observed_label, sample_id). No clean labels, no mask."""

    def __init__(self, images: np.ndarray, y_observed: np.ndarray, sample_ids: np.ndarray) -> None:
        self.images = images
        self.y_observed = y_observed
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        x = torch.as_tensor(self.images[i], dtype=torch.uint8)
        return x, int(self.y_observed[i]), int(self.sample_ids[i])

    @property
    def num_train(self) -> int:
        return len(self.images)


class EvaluationView(Dataset):
    """Training data plus evaluation-only clean labels and noise mask.

    Used exclusively by metrics / detectability / retention code. Training,
    threshold selection, prototype construction, and model selection must never
    read ``clean_labels`` or ``mask`` from this object.
    """

    def __init__(
        self,
        images: np.ndarray,
        y_observed: np.ndarray,
        y_clean: np.ndarray,
        mask: np.ndarray,
        sample_ids: np.ndarray,
    ) -> None:
        self.images = images
        self.y_observed = y_observed
        self.clean_labels = y_clean
        self.mask = mask.astype(bool)
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int, int, int]:
        x = torch.as_tensor(self.images[i], dtype=torch.uint8)
        return x, int(self.y_observed[i]), int(self.clean_labels[i]), int(self.mask[i]), int(self.sample_ids[i])


@dataclass
class NoisyDatasetBundle:
    """Pair of views sharing the same sample ordering (join on sample_id)."""

    train_view: TrainingView
    eval_view: EvaluationView
    num_classes: int
    num_train: int
    num_noisy: int
    registry_entry: dict[str, Any]


def train_eval_split(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic permutation split into (train_idx, val_idx) on the noisy
    training set. The validation split carries noisy labels only (used for
    early stopping / model selection)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    return perm[n_val:], perm[:n_val]


def load_bundle(cfg: dict, raw: Any | None = None) -> NoisyDatasetBundle:
    """Load the raw dataset and build views for a resolved run config.

    ``raw`` may be supplied (e.g. in tests) to avoid downloading CIFAR.
    For instance_dependent noise, pixel features are used to drive flips.
    For the CIFAR-10N real-annotation dataset, the human noise labels come from
    CIFAR-10_human_anno.npz (no synthetic flip; the corruption rate is NEVER
    used as a training input — it exists only in the registry for reporting).
    """
    if cfg["dataset"]["name"] == "cifar10n":
        return build_cifar10n_views(cfg)
    if cfg["dataset"]["name"] == "cifar100n":
        return build_cifar100n_views(cfg)
    if raw is None:
        raw, _ = load_raw_dataset(cfg["dataset"])
    features = None
    if cfg["noise"].get("type") == "instance_dependent":
        imgs, _ = to_numpy_arrays(raw)
        features = imgs.reshape(imgs.shape[0], -1).astype(np.float32)
        features = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-8)
    return build_views(cfg, raw, features=features)


def build_cifar10n_views(cfg: dict) -> NoisyDatasetBundle:
    """CIFAR-10N real-annotation-noise views (human labels, CIFAR-10 images)."""
    return _build_human_noise_views(
        cfg, image_dataset="cifar10", label_loader=load_cifar10n_labels,
        default_subset="aggregate", dataset_name="cifar10n",
    )


def build_cifar100n_views(cfg: dict) -> NoisyDatasetBundle:
    """CIFAR-100N real-annotation-noise views (human labels, CIFAR-100 images)."""
    return _build_human_noise_views(
        cfg, image_dataset="cifar100", label_loader=load_cifar100n_labels,
        default_subset="human", dataset_name="cifar100n",
    )


def _build_human_noise_views(cfg: dict, image_dataset: str, label_loader, default_subset: str,
                             dataset_name: str) -> NoisyDatasetBundle:
    """Build views for a real human-annotation-noise dataset.

    Images come from the base CIFAR dataset; training labels (``y_observed``) are
    the human annotations; clean labels are the expert-verified ones and are used
    for evaluation only.  The realized corruption rate is computed from the
    annotations but is NOT exposed to any training code path (registry only).
    """
    cfg_dataset = cfg["dataset"]
    cfg_noise = cfg["noise"]
    num_classes = int(cfg_dataset["num_classes"])

    raw, _ = load_raw_dataset({**cfg_dataset, "name": image_dataset, "root": None})
    images, y_clean_base = to_numpy_arrays(raw)
    labels = label_loader(cfg_dataset)
    subset = str(cfg_noise.get("subset", default_subset))
    if subset not in labels:
        raise ValueError(f"{dataset_name} subset not found: {subset} (available: {sorted(labels.keys())})")
    y_noisy = labels[subset].astype(int)
    y_clean = labels.get("clean", y_clean_base).astype(int)
    n = images.shape[0]
    if y_noisy.shape[0] != n:
        raise ValueError(f"{dataset_name} label count {y_noisy.shape[0]} != image count {n}")
    mask = (y_noisy != y_clean)
    sample_ids = np.arange(n)

    train_idx, _val_idx = train_eval_split(n, float(cfg_dataset.get("val_frac", 0.1)), int(cfg.get("seed", 0)))
    train_idx = np.sort(train_idx)

    train_view = TrainingView(images=images[train_idx], y_observed=y_noisy[train_idx], sample_ids=sample_ids[train_idx])
    eval_view = EvaluationView(images=images, y_observed=y_noisy, y_clean=y_clean, mask=mask, sample_ids=sample_ids)

    registry_entry = {
        "dataset": dataset_name,
        "noise_type": "real_annotation",
        "noise_rate": None,
        "subset": subset,
        "realized_noise_rate": float(mask.mean()),
        "seed": int(cfg_noise.get("seed", 0)),
        "num_train": int(n),
        "num_noisy": int(mask.sum()),
        "num_classes": num_classes,
        "val_frac": float(cfg_dataset.get("val_frac", 0.1)),
    }
    return NoisyDatasetBundle(
        train_view=train_view, eval_view=eval_view, num_classes=num_classes,
        num_train=int(n), num_noisy=int(mask.sum()), registry_entry=registry_entry,
    )


def _legacy_build_cifar10n_views(cfg: dict) -> NoisyDatasetBundle:
    """Build views for the CIFAR-10N real-annotation-noise setting.

    Images come from CIFAR-10; training labels (y_observed) are the human
    annotation noise labels; clean labels come from CIFAR-10 (evaluation-only).
    The realized corruption rate is computed from the annotation per subset but
    is NOT exposed to any training code path below (registry only).
    """
    cfg_dataset = cfg["dataset"]
    cfg_noise = cfg["noise"]
    num_classes = int(cfg_dataset["num_classes"])

    # CIFAR-10 images + clean labels (CIFAR-10N reuses CIFAR-10 images).
    raw, _ = load_raw_dataset({**cfg_dataset, "name": "cifar10", "root": None})
    images, y_clean_cifar = to_numpy_arrays(raw)
    labels = load_cifar10n_labels(cfg_dataset)
    subset = str(cfg_noise.get("subset", "aggregate"))
    if subset not in labels:
        raise ValueError(f"CIFAR-10N subset not found: {subset} (available: {sorted(labels.keys())})")
    y_noisy = labels[subset].astype(int)
    y_clean = labels.get("clean", y_clean_cifar).astype(int)
    n = images.shape[0]
    if y_noisy.shape[0] != n:
        raise ValueError(f"CIFAR-10N label count {y_noisy.shape[0]} != image count {n}")
    mask = (y_noisy != y_clean)
    sample_ids = np.arange(n)

    train_idx, _val_idx = train_eval_split(n, float(cfg_dataset.get("val_frac", 0.1)), int(cfg.get("seed", 0)))
    train_idx = np.sort(train_idx)

    train_view = TrainingView(images=images[train_idx], y_observed=y_noisy[train_idx], sample_ids=sample_ids[train_idx])
    eval_view = EvaluationView(images=images, y_observed=y_noisy, y_clean=y_clean, mask=mask, sample_ids=sample_ids)

    registry_entry = {
        "dataset": "cifar10n",
        "noise_type": "real_annotation",
        "noise_rate": None,
        "subset": subset,
        "realized_noise_rate": float(mask.mean()),
        "seed": int(cfg_noise.get("seed", 0)),
        "num_train": int(n),
        "num_noisy": int(mask.sum()),
        "num_classes": num_classes,
        "val_frac": float(cfg_dataset.get("val_frac", 0.1)),
    }
    return NoisyDatasetBundle(
        train_view=train_view, eval_view=eval_view, num_classes=num_classes,
        num_train=int(n), num_noisy=int(mask.sum()), registry_entry=registry_entry,
    )


def build_views(cfg: dict, raw: Any, features: np.ndarray | None = None) -> NoisyDatasetBundle:
    """Build TrainingView / EvaluationView from a raw dataset plus noise config.

    ``cfg`` is a resolved experiment config containing ``dataset`` and ``noise``
    sub-configs. ``raw`` is the torchvision CIFAR train set (or any object with
    ``.data`` and ``.targets``). ``features`` is required for
    instance_dependent noise.
    """
    cfg_dataset = cfg["dataset"]
    cfg_noise = cfg["noise"]
    num_classes = int(cfg_dataset["num_classes"])

    images, y_clean = to_numpy_arrays(raw)
    n = images.shape[0]
    sample_ids = np.arange(n)

    result = noise_mod.apply_noise(
        {**cfg_noise, "dataset": cfg_dataset["name"], "num_classes": num_classes},
        y_clean,
        features=features,
    )
    y_noisy = result["y_noisy"]
    mask = result["mask"]

    # Model selection happens on the noisy-label validation split; the eval
    # view keeps the full training set (mask used only for evaluation).
    train_idx, _val_idx = train_eval_split(n, float(cfg_dataset.get("val_frac", 0.1)), int(cfg.get("seed", 0)))
    train_idx = np.sort(train_idx)

    train_view = TrainingView(
        images=images[train_idx],
        y_observed=y_noisy[train_idx],
        sample_ids=sample_ids[train_idx],
    )
    eval_view = EvaluationView(
        images=images,
        y_observed=y_noisy,
        y_clean=y_clean,
        mask=mask,
        sample_ids=sample_ids,
    )

    registry_entry = {
        "dataset": cfg_dataset["name"],
        "noise_type": cfg_noise["type"],
        "noise_rate": float(cfg_noise["rate"]),
        "realized_noise_rate": float(mask.mean()),
        "seed": int(cfg_noise.get("seed", 0)),
        "num_train": int(n),
        "num_noisy": int(mask.sum()),
        "num_classes": num_classes,
        "val_frac": float(cfg_dataset.get("val_frac", 0.1)),
    }
    return NoisyDatasetBundle(
        train_view=train_view,
        eval_view=eval_view,
        num_classes=num_classes,
        num_train=int(n),
        num_noisy=int(mask.sum()),
        registry_entry=registry_entry,
    )
