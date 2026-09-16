"""Noise engine: symmetric, asymmetric/pair-flip, and instance-dependent label noise.

All mechanisms are pure functions of their inputs and a seed: the same inputs +
seed produce the same noise mask. Each mechanism uses its own
``np.random.default_rng(seed)`` and never touches the global RNG. The clean
labels / noise mask are returned here so that ``data.datasets`` can attach them
to the evaluation view only — nothing in this module or the training path
should consume ``mask``.
"""
from __future__ import annotations

import numpy as np

# Well-established CIFAR-10 asymmetric transition map used in the noisy-label
# literature (DivideMix et al.): truck->automobile, bird->airplane,
# deer->horse, cat->dog. Class indices follow torchvision CIFAR-10.
CIFAR10_ASYMMETRIC_MAP: dict[int, int] = {9: 1, 2: 0, 4: 7, 3: 5}

# CIFAR-100: flip within each superclass to the next class inside that block.
def cifar100_superclass_map() -> dict[int, int]:
    return {i: (i // 10) * 10 + (i % 10 + 1) % 10 for i in range(100)}


DEFAULT_ASYMMETRIC_MAPS: dict[str, dict[int, int]] = {
    "cifar10": CIFAR10_ASYMMETRIC_MAP,
    "cifar100": cifar100_superclass_map(),
}


class SymmetricNoise:
    """Each sample independently flips to a uniformly random other class."""

    @staticmethod
    def flip(y: np.ndarray, num_classes: int, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Flip each sample to a uniformly random *other* class with prob ``rate``."""
        y = np.asarray(y)
        n = y.shape[0]
        rng_i = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        mask = rng_i.random(n) < rate
        noisy_y = y.copy()
        n_flip = int(mask.sum())
        if n_flip > 0:
            flip_targets = rng_i.integers(0, num_classes - 1, size=n_flip)
            flip_targets = np.where(flip_targets >= y[mask], flip_targets + 1, flip_targets)
            noisy_y[mask] = flip_targets
        return noisy_y, mask


class AsymmetricNoise:
    """Deterministic class-pair transitions with an average ``rate`` of flips."""

    def __init__(self, transition: dict[int, int] | None = None) -> None:
        self.transition = transition or CIFAR10_ASYMMETRIC_MAP

    def apply(
        self,
        y: np.ndarray,
        num_classes: int,
        rate: float,
        rng: np.random.Generator,
        transition: dict[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        trans = transition or self.transition
        y = np.asarray(y)
        n = y.shape[0]
        rng_i = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        noisy_y = y.copy()
        mask = np.zeros(n, dtype=bool)
        # Within each source class, flip the target fraction of samples to its
        # paired class; iterate classes so the overall rate matches the target.
        for src, dst in trans.items():
            idx = np.where(y == src)[0]
            n_flip = int(round(rate * idx.shape[0]))
            if n_flip > 0 and idx.shape[0] > 0:
                chosen = rng_i.choice(idx, size=n_flip, replace=False)
                noisy_y[chosen] = dst
                mask[chosen] = True
        return noisy_y, mask


class InstanceDependentNoise:
    """Feature-dependent flips: ambiguous instances (low margin to the runner-up
    class) flip more often, and the flip target is the runner-up class.

    This intentionally produces the confounded scenario the project studies:
    noise concentrates where sample quality / representational ambiguity is
    lowest. Flip probability is ``base_rate + alpha * ambiguity`` clipped to
    ``max_rate``; ``alpha`` is calibrated so the *average* flip rate matches
    the target ``rate``. ``base_rate`` is a floor (safe, well-separated samples
    are still flipped at a low baseline rate).
    """

    def __init__(self, base_rate: float = 0.05, max_rate: float = 0.8) -> None:
        self.base_rate = base_rate
        self.max_rate = max_rate
        self._alpha: float | None = None

    def _class_prototypes(self, features: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
        return np.stack(
            [
                features[y == c].mean(axis=0) if (y == c).any() else np.zeros(features.shape[1])
                for c in range(num_classes)
            ]
        )

    def ambiguity(self, features: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
        """Sigmoid-transformed margin to the runner-up class; ambiguous -> ~1."""
        features = np.asarray(features, dtype=np.float32)
        mu = self._class_prototypes(features, y, num_classes)
        f_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        mu_norm = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8)
        sim = f_norm @ mu_norm.T  # (n, num_classes)
        other = sim.copy()
        other[np.arange(sim.shape[0]), y] = -np.inf
        second = other.max(axis=1)
        margin = sim[np.arange(sim.shape[0]), y] - second
        med = np.median(margin)
        std = margin.std() + 1e-8
        z = (margin - med) / std
        return 1.0 / (1.0 + np.exp(z))  # sigmoid(-z)

    def flip_probabilities(self, features: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
        amb = self.ambiguity(features, y, num_classes)
        alpha = self._alpha if self._alpha is not None else 0.0
        return np.clip(self.base_rate + alpha * amb, 0.0, self.max_rate)

    def _calibrate_alpha(
        self, features: np.ndarray, y: np.ndarray, num_classes: int, target_rate: float, seed: int
    ) -> float:
        """Binary-search ``alpha`` so the realized flip rate ≈ target_rate."""
        amb = self.ambiguity(features, y, num_classes)
        hi = (self.max_rate - self.base_rate) / (amb.max() + 1e-8)
        lo = 0.0
        rng = np.random.default_rng(seed)
        rng_seed = int(rng.integers(0, 2**31 - 1))
        for _ in range(10):
            alpha = 0.5 * (lo + hi)
            prob = np.clip(self.base_rate + alpha * amb, 0.0, self.max_rate)
            realized = float(np.mean(np.random.default_rng(rng_seed).random(len(y)) < prob))
            if realized < target_rate:
                lo = alpha
            else:
                hi = alpha
        self._alpha = 0.5 * (lo + hi)
        return self._alpha

    def apply(
        self,
        y: np.ndarray,
        features: np.ndarray,
        num_classes: int,
        rate: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y)
        n = y.shape[0]
        rng_i = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        self._calibrate_alpha(features, y, num_classes, rate, int(rng_i.integers(0, 2**31 - 1)))
        prob = self.flip_probabilities(features, y, num_classes)
        draws = rng_i.random(n)
        mask = draws < prob
        # Flip target = runner-up class.
        mu = self._class_prototypes(features, y, num_classes)
        f_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        mu_norm = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-8)
        sim = f_norm @ mu_norm.T
        other = sim.copy()
        other[np.arange(n), y] = -np.inf
        targets = other.argmax(axis=1)
        noisy_y = y.copy()
        noisy_y[mask] = targets[mask]
        return noisy_y, mask


def apply_noise(
    cfg_noise: dict,
    y_clean: np.ndarray,
    features: np.ndarray | None = None,
) -> dict:
    """Dispatch to the noise mechanism configured in ``cfg_noise``.

    Returns ``{"y_noisy", "mask", "meta"}`` where ``mask`` is boolean
    (True = noisy). ``features`` is required for instance-dependent noise.
    """
    noise_type = cfg_noise["type"]
    rate = float(cfg_noise.get("rate", 0.4))
    seed = int(cfg_noise.get("seed", 0))
    num_classes = int(cfg_noise.get("num_classes", y_clean.max() + 1))
    rng = np.random.default_rng(seed)

    if noise_type == "symmetric":
        noisy_y, mask = SymmetricNoise.flip(y_clean, num_classes, rate, rng)
    elif noise_type == "asymmetric":
        pair_map = cfg_noise.get("pair_map") or DEFAULT_ASYMMETRIC_MAPS.get(
            cfg_noise.get("dataset", "cifar10"), CIFAR10_ASYMMETRIC_MAP
        )
        noisy_y, mask = AsymmetricNoise().apply(y_clean, num_classes, rate, rng, transition=pair_map)
    elif noise_type == "instance_dependent":
        if features is None:
            raise ValueError("instance_dependent noise requires `features`")
        noisy_y, mask = InstanceDependentNoise(
            base_rate=cfg_noise.get("base_rate", 0.05),
            max_rate=cfg_noise.get("max_rate", 0.8),
        ).apply(y_clean, features, num_classes, rate, rng)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return {
        "y_noisy": noisy_y.astype(int),
        "mask": mask.astype(bool),
        "meta": {
            "type": noise_type,
            "rate": rate,
            "realized_rate": float(mask.mean()),
            "seed": seed,
            "num_noisy": int(mask.sum()),
            "num_clean": int((~mask).sum()),
        },
    }
