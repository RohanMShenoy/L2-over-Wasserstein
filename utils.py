"""Shared utilities for the WoW convergence and Bayesian-hyperprior notebooks.

Layout (search for the banner comments):

    1. Style and generic statistics
    2. Ring model (finite-support base ring, samplers, schematic plot)
    3. Exact transport: inner W2, outer W2, WoW to a known target
    4. Experiments 1-2: fixed-C and growing-C convergence
    5. MNIST-family diagnostic (torch is imported lazily, only here)
    6. Bayesian hyperprior study (likelihood, parametric posterior, finite DP)
    7. Results I/O
    8. Plotting (works from the saved/long-format results, so experiment
       notebooks and the plotting notebook share one code path)

Every function takes its parameters explicitly (usually via a ``Config`` from
``config.py``); there is no hidden module-level state apart from the optional
solver back-ends.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import eye, kron, vstack
from scipy.special import logsumexp, ndtr
from tqdm import tqdm

from config import Config

# --------------------------------------------------------------------------- #
# 1. Style and generic statistics
# --------------------------------------------------------------------------- #
BG = "#FFFFFF"
COLORS = cm.viridis(np.linspace(0, 1, 10))


def apply_style() -> None:
    """Common matplotlib style used by every notebook."""
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
    })


def mean_ci(values, axis: int = 1, multiplier: float = 1.96):
    """Mean and normal-approximation CI half-width, reducing over `axis` (repeats)."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=axis)
    n = values.shape[axis]
    if n > 1:
        half = multiplier * values.std(axis=axis, ddof=1) / np.sqrt(n)
    else:
        half = np.zeros_like(mean)
    return mean, half


# --------------------------------------------------------------------------- #
# 2. Ring model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class Ring:
    """Fixed finite-support base ring nu (centred up to float precision)."""
    support: np.ndarray       # (M, 2)
    weights: np.ndarray       # (M,)
    mean: np.ndarray          # (2,)
    radius: float
    jitter: float
    seed: int

    @property
    def size(self) -> int:
        return len(self.support)


@lru_cache(maxsize=None)
def _build_ring(seed: int, radius: float, jitter: float, size: int) -> Ring:
    if size % 2:
        raise ValueError("ring_support_size must be even for antipodal pairing.")
    rng = np.random.default_rng(seed)
    half = size // 2
    paired_jitter = rng.normal(0.0, jitter, size=half)
    radial_jitter = np.concatenate([paired_jitter, paired_jitter])
    theta = 2.0 * np.pi * np.arange(size) / size
    radii = np.maximum(radius + radial_jitter, 0.05)
    support = np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])
    weights = np.full(size, 1.0 / size)
    return Ring(support, weights, weights @ support, radius, jitter, seed)


def build_ring(cfg: Config) -> Ring:
    """Antipodal paired radial jitter => the support has mean zero."""
    return _build_ring(cfg.seed, cfg.ring_radius, cfg.ring_radial_jitter, cfg.ring_support_size)


def sample_ring_points(ring: Ring, centre, n: int, rng) -> np.ndarray:
    """n i.i.d. observations from the translated finite-support ring law."""
    atom_ids = rng.integers(0, ring.size, size=int(n))
    return ring.support[atom_ids] + np.asarray(centre, float)


def sample_class_measure(ring: Ring, centre, n: int, rng) -> dict:
    """Empirical class measure (centred support points + weights) with its true centre."""
    atom_ids = rng.integers(0, ring.size, size=int(n))
    unique_ids, counts = np.unique(atom_ids, return_counts=True)
    return {
        "points": ring.support[unique_ids],   # centred; add `centre` for feature space
        "weights": counts.astype(float) / float(n),
        "center": np.asarray(centre, float),
        "n_observations": int(n),
    }


def sample_hierarchy_at_centres(ring: Ring, centres, n: int, rng) -> list:
    return [sample_class_measure(ring, c, n, rng) for c in np.asarray(centres, float)]


def plot_ring_schematic(cfg: Config, ring: Optional[Ring] = None,
                        n_low: Optional[int] = None, n_high: Optional[int] = None):
    """True ring supports, low-n and high-n empirical observations."""
    ring = ring or build_ring(cfg)
    sizes = cfg.fixed_c_sample_sizes
    n_low = int(sizes[min(2, len(sizes) - 1)]) if n_low is None else n_low
    n_high = int(sizes[max(-2, -len(sizes))]) if n_high is None else n_high
    centres = cfg.fixed_c_centres
    rng = np.random.default_rng(cfg.seed)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 3, figsize=(25, 4.5), gridspec_kw={"wspace": 0.1})
    for k, centre in enumerate(centres):
        support = ring.support + centre
        axes[0].scatter(support[:, 0], support[:, 1], s=4, alpha=0.85, color=cycle[k % len(cycle)])
    axes[0].set_title("True class laws", fontsize=20)

    for ax, n in [(axes[1], n_low), (axes[2], n_high)]:
        for k, centre in enumerate(centres):
            support = ring.support + centre
            points = sample_ring_points(ring, centre, n, rng)
            ax.scatter(support[:, 0], support[:, 1], s=9, alpha=0.071, color=cycle[k % len(cycle)])
            ax.scatter(points[:, 0], points[:, 1], s=20 if n < 100 else 6, alpha=0.8,
                       color=cycle[k % len(cycle)], rasterized=True)
        ax.set_title(fr"Empirical classes: $n={n}$", fontsize=20)

    span = float(np.abs(centres[:, 0]).max()) + 1.0
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-span, span)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Ring classes", fontsize=30, weight="bold")
    return fig


# --------------------------------------------------------------------------- #
# 3. Exact transport
# --------------------------------------------------------------------------- #
try:
    import ot  # POT: optional fast exact EMD
    HAVE_POT = True
except Exception:  # pragma: no cover
    ot = None
    HAVE_POT = False
HAVE_SCIPY_OT = all(x is not None for x in (linear_sum_assignment, linprog, eye, kron, vstack))


def ot_backend() -> str:
    if HAVE_POT:
        return "POT available; SciPy assignment used for equal-weight square problems"
    if HAVE_SCIPY_OT:
        return "SciPy exact assignment/linear programming (no POT needed)"
    warnings.warn("Neither POT nor SciPy OT solvers are available; distances are approximate.")
    return "sliced inner-W2 / greedy outer fallback (approximate)"


@lru_cache(maxsize=None)
def _transport_constraint_matrix(n: int, m: int):
    rows = kron(eye(n, format="csr"), np.ones((1, m)), format="csr")
    cols = kron(np.ones((1, n)), eye(m, format="csr"), format="csr")
    return vstack([rows, cols], format="csr")


def _sqeuclidean_cost(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    return np.maximum(
        np.sum(X * X, axis=1)[:, None] + np.sum(Y * Y, axis=1)[None, :] - 2.0 * X @ Y.T, 0.0)


def _normalise_weights(w, n):
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=float).reshape(-1)
    if len(w) != n or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("Weights must be finite, non-negative, and match the support size.")
    return w / w.sum()


def _weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cdf_mid = np.cumsum(w) - 0.5 * w
    return np.interp(quantiles, cdf_mid, v, left=v[0], right=v[-1])


def _uniform_quantiles_from_projected(projected, quantiles):
    """Equal-weight quantiles. `projected`: (classes, samples, projections) ->
    (classes, len(quantiles), projections), midpoint-CDF interpolation."""
    projected = np.sort(np.asarray(projected), axis=1)
    n = projected.shape[1]
    if n == 1:
        return np.repeat(projected, len(quantiles), axis=1)
    midpoints = (np.arange(n, dtype=float) + 0.5) / n
    q = np.asarray(quantiles, dtype=float)
    hi = np.searchsorted(midpoints, q, side="right")
    lo = np.clip(hi - 1, 0, n - 1)
    hi = np.clip(hi, 0, n - 1)
    denominator = midpoints[hi] - midpoints[lo]
    alpha = np.divide(q - midpoints[lo], denominator, out=np.zeros_like(q), where=denominator > 0)
    alpha = alpha.reshape(1, -1, 1)
    result = (1.0 - alpha) * projected[:, lo, :] + alpha * projected[:, hi, :]
    result[:, q <= midpoints[0], :] = projected[:, :1, :]
    result[:, q >= midpoints[-1], :] = projected[:, -1:, :]
    return result


def _sliced_w2_squared(X, a, Y, b, n_projections=128, seed=13):
    """Approximate inner W2^2 by 1-D projections (fallback / MNIST diagnostic)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    a, b = _normalise_weights(a, len(X)), _normalise_weights(b, len(Y))
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_projections, X.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    q = (np.arange(128) + 0.5) / 128.0
    px, py = X @ directions.T, Y @ directions.T
    if np.allclose(a, 1.0 / len(a)) and np.allclose(b, 1.0 / len(b)):
        qx = _uniform_quantiles_from_projected(px[None], q)[0]
        qy = _uniform_quantiles_from_projected(py[None], q)[0]
        return float(np.mean((qx - qy) ** 2))
    vals = [np.mean((_weighted_quantile(px[:, k], a, q) - _weighted_quantile(py[:, k], b, q)) ** 2)
            for k in range(n_projections)]
    return float(np.mean(vals))


def _transport_lp(cost, a, b):
    n, m = cost.shape
    result = linprog(np.asarray(cost, float).ravel(), A_eq=_transport_constraint_matrix(n, m),
                     b_eq=np.concatenate([a, b]), bounds=(0.0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"Exact transportation LP failed: {result.message}")
    return float(result.fun)


def _greedy_transport_fallback(cost, a, b):
    ra, rb = a.copy(), b.copy()
    total = 0.0
    for i, j in sorted(np.ndindex(cost.shape), key=lambda ij: cost[ij]):
        amount = min(ra[i], rb[j])
        if amount > 0:
            total += amount * cost[i, j]
            ra[i] -= amount
            rb[j] -= amount
    return float(total)


def weighted_w2_squared(X, a, Y, b):
    """W2^2 between finite measures given as (support, weights)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must be 2-D arrays with the same feature dimension.")
    if len(X) == 0 or len(Y) == 0:
        raise ValueError("Measures must have non-empty supports.")
    a, b = _normalise_weights(a, len(X)), _normalise_weights(b, len(Y))
    cost = _sqeuclidean_cost(X, Y)
    uniform = np.allclose(a, 1.0 / len(a)) and np.allclose(b, 1.0 / len(b))
    if HAVE_SCIPY_OT and len(X) == len(Y) and uniform:
        rows, cols = linear_sum_assignment(cost)
        return float(cost[rows, cols].mean())
    if HAVE_POT:
        return float(ot.emd2(a, b, cost))
    if HAVE_SCIPY_OT:
        return _transport_lp(cost, a, b)
    return _sliced_w2_squared(X, a, Y, b)


def outer_w2_squared(cost, a, b):
    """Outer W2^2 over class distributions with class-to-class costs as ground costs."""
    cost = np.asarray(cost, dtype=float)
    a, b = _normalise_weights(a, cost.shape[0]), _normalise_weights(b, cost.shape[1])
    uniform = np.allclose(a, 1.0 / len(a)) and np.allclose(b, 1.0 / len(b))
    if HAVE_SCIPY_OT and cost.shape[0] == cost.shape[1] and uniform:
        rows, cols = linear_sum_assignment(cost)
        return float(cost[rows, cols].mean())
    if HAVE_POT:
        return float(ot.emd2(a, b, cost))
    if HAVE_SCIPY_OT:
        return _transport_lp(cost, a, b)
    return _greedy_transport_fallback(cost, a, b)


def wow_between_hierarchies(A_measures, B_measures, inner_metric="exact", n_projections=32, seed=13):
    """Outer W2 between two hierarchies, with exact or sliced inner costs."""
    if not A_measures or not B_measures:
        raise ValueError("Both hierarchies must contain at least one class measure.")
    if inner_metric not in {"exact", "sliced"}:
        raise ValueError("inner_metric must be 'exact' or 'sliced'.")
    cost = np.empty((len(A_measures), len(B_measures)))
    for i, am in enumerate(A_measures):
        for j, bm in enumerate(B_measures):
            if inner_metric == "exact":
                cost[i, j] = weighted_w2_squared(am["points"], am["weights"], bm["points"], bm["weights"])
            else:
                cost[i, j] = _sliced_w2_squared(am["points"], am["weights"], bm["points"],
                                                bm["weights"], n_projections=n_projections, seed=seed)
    oa = np.full(len(A_measures), 1.0 / len(A_measures))
    ob = np.full(len(B_measures), 1.0 / len(B_measures))
    return float(np.sqrt(max(outer_w2_squared(cost, oa, ob), 0.0)))


def measures_to_feature_hierarchy(arrays):
    """(C, n, d) feature arrays -> equally weighted empirical class measures."""
    arrays = np.asarray(arrays, dtype=float)
    return [{"points": arrays[c], "weights": np.full(arrays.shape[1], 1.0 / arrays.shape[1]),
             "class_type": None, "n_observations": arrays.shape[1]} for c in range(arrays.shape[0])]


# --- translated-ring identities (exact WoW to known targets) ------------------
def _prepare_class_measure(measure, ring: Ring):
    base_w2_sq = weighted_w2_squared(measure["points"], measure["weights"], ring.support, ring.weights)
    mean_base = np.average(measure["points"], axis=0, weights=measure["weights"])
    return base_w2_sq, mean_base - ring.mean, measure["center"]


def _inner_cost_to_true_ring(prepared, target_centre):
    """Exact W2^2 to a true ring law via the translation identity."""
    base_w2_sq, mean_offset, source_centre = prepared
    delta = np.asarray(source_centre) - np.asarray(target_centre)
    return float(max(base_w2_sq + 2.0 * np.dot(delta, mean_offset) + np.dot(delta, delta), 0.0))


def wow_to_fixed_target(measures, ring: Ring, target_centres):
    """Exact WoW distance to the finite hierarchy of rings centred at `target_centres`."""
    target_centres = np.asarray(target_centres, dtype=float)
    if not measures:
        raise ValueError("The source hierarchy must contain at least one class measure.")
    prepared = [_prepare_class_measure(m, ring) for m in measures]
    inner = np.array([[_inner_cost_to_true_ring(m, t) for t in target_centres] for m in prepared])
    p = np.full(len(measures), 1.0 / len(measures))
    q = np.full(len(target_centres), 1.0 / len(target_centres))
    return float(np.sqrt(max(outer_w2_squared(inner, p, q), 0.0)))


def w2_to_uniform_squared(support, weights=None, limit: float = 2.0):
    """Exact W2^2 between a discrete 1-D measure and Uniform[-limit, limit].

    `weights=None` gives the equally weighted empirical measure on `support`.
    `weights` of shape (n_draws, K) is a batch sharing `support`; the result then
    has shape (n_draws,).  On each mass cell [u_{j-1}, u_j) the quantile of the
    discrete law is constant, so

        W2^2 = sum_j w_j (a_j - q_j)^2 + (2L)^2 w_j^3 / 12,   q_j = -L + 2L (u_{j-1}+u_j)/2 .

    With w_j = 1/C this reduces to the empirical-to-uniform formula
    mean (x_(i) - bin mean)^2 + L^2 / (3 C^2).
    """
    support = np.asarray(support, dtype=float).reshape(-1)
    if weights is None:
        weights = np.full((1, support.size), 1.0 / support.size)
        squeeze = True
    else:
        weights = np.asarray(weights, dtype=float)
        squeeze = weights.ndim == 1
        if squeeze:
            weights = weights[None, :]
    if weights.ndim != 2 or weights.shape[1] != support.size:
        raise ValueError("weights must have shape (n_draws, len(support)).")
    if np.any(weights < 0.0):
        raise ValueError("Weights must be non-negative.")
    row_sums = weights.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Every weight vector must have positive total mass.")
    order = np.argsort(support)
    s = support[order]
    w = weights[:, order] / row_sums
    upper = np.cumsum(w, axis=1)
    lower = upper - w
    q_mid = -limit + 2.0 * limit * 0.5 * (lower + upper)
    cost = np.sum(w * (s[None, :] - q_mid) ** 2 + (2.0 * limit) ** 2 * w ** 3 / 12.0, axis=1)
    cost = np.maximum(cost, 0.0)
    return float(cost[0]) if squeeze else cost


def w2_to_uniform(support, weights=None, limit: float = 2.0):
    """Exact W2 (square root of `w2_to_uniform_squared`)."""
    out = np.sqrt(w2_to_uniform_squared(support, weights, limit))
    return float(out) if np.ndim(out) == 0 else out


def wow_to_continuous_population(measures, ring: Ring, half_width: float):
    """Exact WoW distance to P = Law(mu_A), A ~ Uniform([-L, L]), for translated rings.

    With e = W2^2(nu_hat, nu), m = mean(nu_hat) - mean(nu), z = centre + m:
    W2^2(nu_hat + centre, nu + a) = (e - |m|^2) + |z - a|^2, so the outer OT splits into
    the mean residual, a 1-D empirical-to-uniform term in z_x and the vertical offset z_y.
    """
    if not measures:
        raise ValueError("The source hierarchy must contain at least one class measure.")
    prepared = [_prepare_class_measure(m, ring) for m in measures]
    base_costs = np.array([p[0] for p in prepared])
    offsets = np.stack([p[1] for p in prepared])
    centres = np.stack([p[2] for p in prepared])
    residuals = np.maximum(base_costs - np.sum(offsets ** 2, axis=1), 0.0)
    z = centres + offsets
    outer_x = w2_to_uniform_squared(z[:, 0], None, half_width)
    outer_y = float(np.mean(z[:, 1] ** 2))
    return float(np.sqrt(max(np.mean(residuals) + outer_x + outer_y, 0.0)))


# --------------------------------------------------------------------------- #
# 4. Experiments 1-2: fixed-C and growing-C convergence
# --------------------------------------------------------------------------- #
FRAME_COLUMNS = ["experiment", "C", "n_per_class", "repetition", "distance", "seed"]


def _frame(experiment, C, n_per_class, repetition, distance, seed, **extra) -> pd.DataFrame:
    df = pd.DataFrame({"experiment": experiment, "C": C, "n_per_class": n_per_class,
                       "repetition": repetition, "distance": distance, "seed": seed})
    for k, v in extra.items():
        df[k] = v
    return df


def run_fixed_c(cfg: Config, ring: Optional[Ring] = None) -> pd.DataFrame:
    """Fixed C: WoW(P_hat_{C,n}, P_C) as n grows."""
    ring = ring or build_ring(cfg)
    centres = cfg.fixed_c_centres
    rng = np.random.default_rng(cfg.seed)
    n_grid, rows = cfg.fixed_c_sample_sizes, []
    for n in tqdm(n_grid, desc="Fixed-C convergence"):
        for rep in range(cfg.fixed_c_repeats):
            hier = sample_hierarchy_at_centres(ring, centres, int(n), rng)
            rows.append((int(n), rep, wow_to_fixed_target(hier, ring, centres)))
    n_col, rep_col, dist = map(np.array, zip(*rows))
    return _frame("fixed_c", len(centres), n_col, rep_col, dist, cfg.seed)


def run_growing_c(cfg: Config, ring: Optional[Ring] = None) -> pd.DataFrame:
    """Growing C: joint (C, n(C)) growth, fixed-n control, and centre-only diagnostic."""
    ring = ring or build_ring(cfg)
    L = cfg.centre_limit
    rng = np.random.default_rng(cfg.seed)
    parts = {"joint_growth": [], "fixed_n_control": [], "centre_only": []}
    for C, n_joint in tqdm(list(zip(cfg.meta_c_values, cfg.meta_n_joint)), desc="Growing-C convergence"):
        for rep in range(cfg.meta_repeats):
            centres = np.column_stack([rng.uniform(-L, L, size=C), np.zeros(C)])
            joint = sample_hierarchy_at_centres(ring, centres, int(n_joint), rng)
            parts["joint_growth"].append((int(C), int(n_joint), rep,
                                          wow_to_continuous_population(joint, ring, L)))
            parts["centre_only"].append((int(C), np.nan, rep,
                                         float(np.sqrt(w2_to_uniform_squared(centres[:, 0], None, L)))))
            fixed = sample_hierarchy_at_centres(ring, centres, cfg.meta_n_fixed, rng)
            parts["fixed_n_control"].append((int(C), cfg.meta_n_fixed, rep,
                                             wow_to_continuous_population(fixed, ring, L)))
    frames = []
    for name, rows in parts.items():
        C_, n_, r_, d_ = map(np.array, zip(*rows))
        frames.append(_frame(name, C_, n_, r_, d_, cfg.seed))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 5. MNIST-family diagnostic (torch imported lazily)
# --------------------------------------------------------------------------- #
def _project_feature_classes(features, directions):
    """Project (classes, samples, features) onto shared random directions."""
    features = np.asarray(features, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    return np.einsum("cnd,pd->cnp", features, directions, optimize=True)


def wow_from_sliced_quantiles(A_quantiles, B_quantiles):
    """Outer empirical W2 with vectorised sliced-W2 inner costs."""
    delta = A_quantiles[:, None, :, :] - B_quantiles[None, :, :, :]
    cost = np.mean(delta * delta, axis=(2, 3))
    oa = np.full(cost.shape[0], 1.0 / cost.shape[0])
    ob = np.full(cost.shape[1], 1.0 / cost.shape[1])
    return float(np.sqrt(max(outer_w2_squared(cost, oa, ob), 0.0)))


def load_image_banks(cfg: Config) -> dict:
    """Sample train banks and two disjoint held-out reference banks per repeat/class.

    Requires torch + torchvision (imported here so the rest of utils does not).
    """
    import torch
    import torch.nn.functional as F
    from torchvision import datasets, transforms

    classes = {"MNIST": datasets.MNIST, "FashionMNIST": datasets.FashionMNIST,
               "KMNIST": datasets.KMNIST, "USPS": datasets.USPS}
    if cfg.mnist_variant not in classes:
        raise ValueError(f"mnist_variant must be one of {list(classes)}")
    size = cfg.mnist_image_size
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    cls = classes[cfg.mnist_variant]
    train_ds = cls(root=cfg.mnist_data_root, train=True, download=True, transform=tf)
    test_ds = cls(root=cfg.mnist_data_root, train=False, download=True, transform=tf)
    train_labels = np.asarray(train_ds.targets, dtype=int)
    test_labels = np.asarray(test_ds.targets, dtype=int)
    class_ids = np.sort(np.intersect1d(np.unique(train_labels), np.unique(test_labels)))
    min_train = min(np.sum(train_labels == c) for c in class_ids)
    min_test = min(np.sum(test_labels == c) for c in class_ids)
    reference_n = min(cfg.mnist_reference_size, min_test // 2)
    sizes = [int(n) for n in cfg.mnist_sample_sizes if n <= min_train and n <= reference_n]
    if not sizes:
        raise ValueError("No requested mnist sample size fits the train counts and reference size.")
    max_n = max(sizes)
    feature_dim = size ** 2

    def load_bank(dataset, idx):
        idx = np.asarray(idx, dtype=int)
        raw_data = getattr(dataset, "data", None)
        if raw_data is not None:
            try:
                raw = raw_data if torch.is_tensor(raw_data) else torch.as_tensor(np.asarray(raw_data))
                if raw.ndim == 3:
                    flat = torch.as_tensor(idx.reshape(-1), dtype=torch.long)
                    batch = raw[flat].unsqueeze(1)
                    was_int = not batch.is_floating_point()
                    batch = batch.to(dtype=torch.float32)
                    if was_int or (batch.numel() and batch.max().item() > 1.0):
                        batch = batch / 255.0
                    out = F.interpolate(batch, size=(size, size), mode="bilinear",
                                        align_corners=False, antialias=True)
                    return out[:, 0].reshape(*idx.shape, -1).cpu().numpy().astype(np.float32)
            except (TypeError, ValueError, RuntimeError, IndexError):
                pass
        bank = np.empty((*idx.shape, feature_dim), dtype=np.float32)
        for r in range(idx.shape[0]):
            for cpos in range(idx.shape[1]):
                for s, i in enumerate(idx[r, cpos]):
                    img, _ = dataset[int(i)]
                    bank[r, cpos, s] = np.asarray(img, dtype=np.float32).reshape(-1)
        return bank

    rng = np.random.default_rng(cfg.seed)
    train_idx = np.empty((cfg.mnist_repeats, len(class_ids), max_n), dtype=int)
    ref_idx = np.empty((cfg.mnist_repeats, len(class_ids), 2 * reference_n), dtype=int)
    for cpos, cid in enumerate(tqdm(class_ids, desc="Selecting image samples")):
        train_c = np.flatnonzero(train_labels == cid)
        test_c = np.flatnonzero(test_labels == cid)
        for rep in range(cfg.mnist_repeats):
            train_idx[rep, cpos] = rng.choice(train_c, size=max_n, replace=False)
            ref_idx[rep, cpos] = rng.permutation(test_c)[:2 * reference_n]

    train_bank = load_bank(train_ds, train_idx)
    reference = load_bank(test_ds, ref_idx)
    return {"train_bank": train_bank,
            "reference_a_bank": reference[:, :, :reference_n, :],
            "reference_b_bank": reference[:, :, reference_n:, :],
            "class_ids": class_ids, "sizes": sizes,
            "reference_n": reference_n, "feature_dim": feature_dim}


def run_mnist(cfg: Config, banks: dict) -> pd.DataFrame:
    """Train-to-reference discrepancy (sliced inner W2, exact outer) and reference-split noise."""
    sizes, reference_n = banks["sizes"], banks["reference_n"]
    feature_dim = banks["feature_dim"]
    q = (np.arange(cfg.mnist_quantile_points, dtype=float) + 0.5) / cfg.mnist_quantile_points
    rng = np.random.default_rng(cfg.seed)
    directions = rng.normal(size=(cfg.mnist_sw_projections, feature_dim)).astype(np.float32)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)

    train_rows, ref_rows = [], []
    for rep in tqdm(range(cfg.mnist_repeats), desc=f"{cfg.mnist_variant} reference splits"):
        proj_train = _project_feature_classes(banks["train_bank"][rep], directions)
        proj_ref_a = _project_feature_classes(banks["reference_a_bank"][rep], directions)
        proj_ref_b = _project_feature_classes(banks["reference_b_bank"][rep], directions)
        qa = _uniform_quantiles_from_projected(proj_ref_a, q)
        qb = _uniform_quantiles_from_projected(proj_ref_b, q)
        ref_rows.append((reference_n, rep, wow_from_sliced_quantiles(qa, qb)))
        for n in sizes:
            qt = _uniform_quantiles_from_projected(proj_train[:, :n, :], q)
            train_rows.append((n, rep, wow_from_sliced_quantiles(qt, qa)))

    n_c = len(banks["class_ids"])
    frames = []
    for name, rows in [("train_to_reference", train_rows), ("reference_split", ref_rows)]:
        n_, r_, d_ = map(np.array, zip(*rows))
        frames.append(_frame(name, n_c, n_, r_, d_, cfg.seed, variant=cfg.mnist_variant))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 6. Bayesian hyperprior study
# --------------------------------------------------------------------------- #
try:
    from numba import njit
    NUMBA_AVAILABLE = True
    NUMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    njit = None
    NUMBA_AVAILABLE = False
    NUMBA_IMPORT_ERROR = repr(exc)

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _gibbs_assignment_counts_numba(class_loglik, log_weights, uniforms):
        """Assignment counts from stable categorical probabilities."""
        C, K = class_loglik.shape
        counts = np.zeros(K, dtype=np.int64)
        for c in range(C):
            m = -np.inf
            for j in range(K):
                v = class_loglik[c, j] + log_weights[j]
                if v > m:
                    m = v
            norm = 0.0
            for j in range(K):
                norm += np.exp(class_loglik[c, j] + log_weights[j] - m)
            threshold = uniforms[c] * norm
            cumulative = 0.0
            selected = K - 1
            for j in range(K):
                cumulative += np.exp(class_loglik[c, j] + log_weights[j] - m)
                if threshold <= cumulative:
                    selected = j
                    break
            counts[selected] += 1
        return counts
else:
    _gibbs_assignment_counts_numba = None


def generate_bayes_replicates(cfg: Config, ring: Optional[Ring] = None):
    """Nested replicates: the largest (C, n) bank is generated once per repetition,
    smaller settings use prefixes.  Returns (latent_centres_bank, data_bank)."""
    ring = ring or build_ring(cfg)
    L = cfg.centre_limit
    max_c, max_n = int(cfg.bayes_c_values.max()), int(max(cfg.bayes_n_values))
    rng = np.random.default_rng(cfg.seed)
    latent = np.empty((cfg.bayes_repeats, max_c))
    data = np.empty((cfg.bayes_repeats, max_c, max_n, 2))
    for rep in tqdm(range(cfg.bayes_repeats), desc="Generating ring replicates"):
        latent[rep] = rng.uniform(-L, L, size=max_c)
        atom_ids = rng.integers(0, ring.size, size=(max_c, max_n))
        X = ring.support[atom_ids].copy()
        X[:, :, 0] += latent[rep][:, None]
        data[rep] = X
    return latent, data


def class_loglikelihoods_chunked(X, cfg: Config, n_values=None, centres=None, chunk_size=None):
    """Class-by-centre log-likelihoods under the smooth truncated-Normal radial working
    model, accumulated over observation chunks (no C x n x K tensor).  Returns
    (C, len(n_values), K); n_values must be non-decreasing prefixes."""
    n_values = np.asarray(cfg.bayes_n_values if n_values is None else n_values, dtype=int)
    centres = np.asarray(cfg.bayes_centres if centres is None else centres, dtype=float)
    chunk_size = cfg.bayes_loglik_chunk if chunk_size is None else chunk_size
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[2] != 2:
        raise ValueError("X must have shape (C, n, 2).")
    if n_values.ndim != 1 or len(n_values) == 0:
        raise ValueError("n_values must be a non-empty one-dimensional array.")
    if np.any(n_values <= 0) or np.any(n_values > X.shape[1]):
        raise ValueError("Every requested n must be between 1 and X.shape[1].")
    if np.any(np.diff(n_values) < 0):
        raise ValueError("n_values must be in non-decreasing order.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    R, sig = cfg.ring_radius, cfg.ring_radial_jitter
    C, K = X.shape[0], centres.size
    result = np.empty((C, len(n_values), K))
    cumulative = np.zeros((C, K))
    log_norm = np.log(sig * np.sqrt(2.0 * np.pi)) + np.log(ndtr(R / sig))
    start = 0
    for ni, target in enumerate(n_values):
        while start < target:
            stop = min(start + chunk_size, int(target))
            Xc = X[:, start:stop, :]
            dx = Xc[:, :, 0, None] - centres[None, None, :]
            dy = Xc[:, :, 1, None]
            radius = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-14)
            z = (radius - R) / sig
            cumulative += (-0.5 * z * z - log_norm - np.log(2.0 * np.pi * radius)).sum(axis=1)
            start = stop
        result[:, ni, :] = cumulative
    return result


def check_translation_isometry(ring: Ring, shift=(1.3, 0.0)) -> None:
    """Verify numerically that the common-noise coupling attains W2^2 = |shift|^2."""
    shift = np.asarray(shift, float)
    cost = np.mean(np.sum((ring.support - (ring.support + shift)) ** 2, axis=1))
    print(f"Common-noise coupling squared cost: {cost:.6f}")
    print(f"Theoretical W2 squared:             {float(shift @ shift):.6f}")


def neighbourhood_mass(probabilities, distances, epsilons):
    return np.array([probabilities[distances < eps].sum() for eps in epsilons])


@dataclass(frozen=True, eq=False)
class ParamFamily:
    """Exponential-tilt family w_j(theta) ∝ exp{theta (a_j/L)^2} on the centre grid."""
    theta_grid: np.ndarray
    log_w: np.ndarray            # (T, K)
    w: np.ndarray                # (T, K)
    log_prior_mass: np.ndarray   # (T,)
    prior_mass: np.ndarray       # (T,)
    wow_distances: np.ndarray    # (T,) exact WoW of each grid population to the truth
    prior_ball_mass: np.ndarray  # (E,)
    grid_floor: float            # distance of the equal-weight midpoint grid to Uniform


def build_param_family(cfg: Config) -> ParamFamily:
    centres, L = cfg.bayes_centres, cfg.centre_limit
    theta = cfg.bayes_theta_grid
    spread = (centres / L) ** 2
    log_w = theta[:, None] * spread[None, :]
    log_w -= logsumexp(log_w, axis=1, keepdims=True)
    w = np.exp(log_w)

    quad = np.ones_like(theta)
    quad[[0, -1]] = 0.5
    quad /= quad.sum()
    log_prior = (-0.5 * ((theta - cfg.bayes_theta_prior_mean) / cfg.bayes_theta_prior_sd) ** 2
                 - np.log(cfg.bayes_theta_prior_sd * np.sqrt(2.0 * np.pi)) + np.log(quad))
    log_prior -= logsumexp(log_prior)
    prior = np.exp(log_prior)

    dist = w2_to_uniform(centres, w, L)
    eps = np.asarray(cfg.bayes_epsilons)
    floor = w2_to_uniform(centres, np.full(centres.size, 1.0 / centres.size), L)
    ball = neighbourhood_mass(prior, dist, eps)
    return ParamFamily(theta, log_w, w, log_prior, prior, dist, ball, floor)


def parametric_posterior(class_loglik, family: ParamFamily, class_chunk: int = 32):
    """Posterior over the theta grid; chunked over classes to bound memory."""
    C = class_loglik.shape[0]
    ll = np.zeros(len(family.theta_grid))
    for s in range(0, C, class_chunk):
        chunk = class_loglik[s:min(s + class_chunk, C)]
        ll += logsumexp(chunk[:, None, :] + family.log_w[None, :, :], axis=2).sum(axis=0)
    logp = family.log_prior_mass + ll
    logp -= logsumexp(logp)
    return np.exp(logp)


def dp_base_weights(cfg: Config) -> np.ndarray:
    w = np.exp(-0.5 * (cfg.bayes_centres / cfg.bayes_dp_base_sd) ** 2)
    return w / w.sum()


def sample_finite_dp_posterior(class_loglik, rng, cfg: Config):
    """Gibbs draws of the finite-support DP weights (numba kernel when available)."""
    class_loglik = np.ascontiguousarray(class_loglik, dtype=np.float64)
    C, K = class_loglik.shape
    base = cfg.bayes_dp_alpha * dp_base_weights(cfg)
    weights = rng.dirichlet(base)
    saved = np.empty((cfg.bayes_dp_draws, K))
    total = cfg.bayes_dp_burn_in + cfg.bayes_dp_draws * cfg.bayes_dp_thin
    out = 0
    for it in range(total):
        uniforms = rng.random(C)
        if NUMBA_AVAILABLE:
            counts = _gibbs_assignment_counts_numba(
                class_loglik, np.log(np.maximum(weights, 1e-300)), uniforms)
        else:
            logc = class_loglik + np.log(np.maximum(weights, 1e-300))[None, :]
            logc -= logsumexp(logc, axis=1, keepdims=True)
            cum = np.cumsum(np.exp(logc), axis=1)
            assign = np.minimum((uniforms[:, None] > cum).sum(axis=1), K - 1)
            counts = np.bincount(assign, minlength=K)
        weights = rng.dirichlet(base + counts)
        if it >= cfg.bayes_dp_burn_in and (it - cfg.bayes_dp_burn_in) % cfg.bayes_dp_thin == 0:
            saved[out] = weights
            out += 1
    return saved


def dp_prior_draws(cfg: Config) -> dict:
    """Prior draws of the finite-DP weights and their exact WoW distances to the truth."""
    rng = np.random.default_rng(cfg.seed + 9876)
    weights = rng.dirichlet(cfg.bayes_dp_alpha * dp_base_weights(cfg), size=cfg.bayes_dp_prior_draws)
    dist = w2_to_uniform(cfg.bayes_centres, weights, cfg.centre_limit)
    ball = np.array([np.mean(dist < e) for e in cfg.bayes_epsilons])
    return {"weights": weights, "distances": dist, "ball_mass": ball}


_FINGERPRINT_EXCLUDE = {"bayes_loglik_chunk", "bayes_class_chunk",
                        "bayes_checkpoint_every", "bayes_save_data_bank"}


def bayes_fingerprint(cfg: Config) -> str:
    """Settings that must match for a checkpoint to be resumable."""
    d = cfg.to_dict()
    keep = {k: v for k, v in d.items()
            if (k.startswith("bayes_") and k not in _FINGERPRINT_EXCLUDE)
            or k in {"seed", "ring_radius", "ring_radial_jitter", "ring_support_size", "centre_limit"}}
    return json.dumps(keep, sort_keys=True)


def run_bayes_fits(cfg: Config, data_bank, family: ParamFamily, out_dir) -> dict:
    """Fit both hyperpriors for every (repetition, C, n); resumable per-repetition checkpoints.

    Returns {"param_posterior_probs": (R, nC, nN, T), "dp_distance_draws": (R, nC, nN, D)}.
    """
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    C_values, N_values = cfg.bayes_c_values, np.asarray(cfg.bayes_n_values, dtype=int)
    R, nC, nN = cfg.bayes_repeats, len(C_values), len(N_values)
    T, D = len(cfg.bayes_theta_grid), cfg.bayes_dp_draws
    centres, L = cfg.bayes_centres, cfg.centre_limit
    fingerprint = bayes_fingerprint(cfg)
    per_rep = nC * nN

    probs_all = np.empty((R, nC, nN, T))
    dist_all = np.empty((R, nC, nN, D))

    def write_ckpt(path, rep, completed, probs, dists):
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, repetition=np.array(rep), completed=np.array(completed),
                            fingerprint=np.array(fingerprint), probs=probs, dists=dists)
        tmp.replace(path)

    with tqdm(total=R * per_rep, desc="Bayesian posterior fits") as bar:
        for rep in range(R):
            path = ckpt_dir / f"repetition_{rep:02d}.npz"
            probs = np.full((nC, nN, T), np.nan)
            dists = np.full((nC, nN, D), np.nan)
            completed = 0
            if path.exists():
                try:
                    with np.load(path) as ck:
                        ok = (str(ck["fingerprint"]) == fingerprint and int(ck["repetition"]) == rep
                              and ck["probs"].shape == probs.shape and ck["dists"].shape == dists.shape)
                        if ok:
                            probs, dists = ck["probs"].copy(), ck["dists"].copy()
                            completed = min(max(int(ck["completed"]), 0), per_rep)
                        else:
                            print(f"Ignoring incompatible checkpoint: {path.name}")
                except Exception as exc:
                    print(f"Could not read checkpoint {path.name}: {exc}")
            bar.update(completed)

            if completed < per_rep:
                loglik_by_n = class_loglikelihoods_chunked(data_bank[rep], cfg)
                for flat in range(completed, per_rep):
                    ci, ni = divmod(flat, nN)
                    C, n = int(C_values[ci]), int(N_values[ni])
                    ll = loglik_by_n[:C, ni, :]
                    probs[ci, ni] = parametric_posterior(ll, family, cfg.bayes_class_chunk)
                    rng = np.random.default_rng(cfg.seed + 100000 * (rep + 1) + 1000 * C + n)
                    w_draws = sample_finite_dp_posterior(ll, rng, cfg)
                    dists[ci, ni] = w2_to_uniform(centres, w_draws, L)
                    completed = flat + 1
                    bar.update(1)
                    if completed % cfg.bayes_checkpoint_every == 0 or completed == per_rep:
                        write_ckpt(path, rep, completed, probs, dists)
            probs_all[rep], dist_all[rep] = probs, dists
    print(f"Posterior fitting complete. Checkpoints are in: {ckpt_dir.resolve()}")
    return {"param_posterior_probs": probs_all, "dp_distance_draws": dist_all}


def bayes_summaries(res: dict) -> dict:
    """Derived posterior summaries from the saved arrays.

    param_mass / dp_mass: (R, nC, nN, E) posterior mass of WoW epsilon-balls;
    param_mean_distance / dp_mean_distance: (R, nC, nN).
    """
    eps = np.asarray(res["epsilons"])
    masks = (np.asarray(res["theta_wow_distances"])[None, :] < eps[:, None]).astype(float)  # (E, T)
    probs, draws = res["param_posterior_probs"], res["dp_distance_draws"]
    return {
        "param_mass": probs @ masks.T,
        "dp_mass": (draws[..., None] < eps).mean(axis=-2),
        "param_mean_distance": probs @ np.asarray(res["theta_wow_distances"]),
        "dp_mean_distance": draws.mean(axis=-1),
    }


# --------------------------------------------------------------------------- #
# 7. Results I/O   (results/<experiment>/<artifact>.<ext>)
# --------------------------------------------------------------------------- #
def experiment_dir(cfg: Config, experiment: str) -> Path:
    path = cfg.results_dir / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(cfg: Config, directory) -> Path:
    path = Path(directory) / "config.json"
    path.write_text(json.dumps(cfg.to_dict(), indent=2))
    return path


def convergence_path(cfg: Config) -> Path:
    return cfg.results_dir / "convergence" / "convergence.csv"


def mnist_path(cfg: Config, variant: Optional[str] = None) -> Path:
    return cfg.results_dir / "mnist" / f"{(variant or cfg.mnist_variant).lower()}.csv"


def bayes_dir(cfg: Config) -> Path:
    return cfg.results_dir / "bayes"


def save_convergence(cfg: Config, *frames: pd.DataFrame) -> Path:
    """Write fixed-C and/or growing-C frames into one long-format CSV.

    Existing rows for the experiments being written are replaced; others are kept,
    so the two convergence experiments can be re-run independently.
    """
    path = convergence_path(cfg)
    experiment_dir(cfg, "convergence")
    new = pd.concat(frames, ignore_index=True)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["experiment"].isin(new["experiment"].unique())], new],
                        ignore_index=True)
    new.to_csv(path, index=False)
    save_config(cfg, path.parent)
    print(f"Saved {len(new)} rows to {path}")
    return path


def save_mnist(cfg: Config, df: pd.DataFrame) -> Path:
    path = mnist_path(cfg)
    experiment_dir(cfg, "mnist")
    df.to_csv(path, index=False)
    save_config(cfg, path.parent)
    print(f"Saved {len(df)} rows to {path}")
    return path


def _flat_index(shape):
    return [a.ravel() for a in np.indices(shape)]


def save_bayes_results(cfg: Config, data_bank, latent_centres_bank, fits: dict,
                       family: ParamFamily, dp_prior: dict) -> Path:
    """Raw .npz archive, replicate-level CSVs and the configuration."""
    out = experiment_dir(cfg, "bayes")
    C_values, N_values = cfg.bayes_c_values, np.asarray(cfg.bayes_n_values, dtype=int)
    arrays = dict(
        latent_centres_bank=latent_centres_bank,
        param_posterior_probs=fits["param_posterior_probs"],
        dp_distance_draws=fits["dp_distance_draws"],
        dp_prior_weight_draws=dp_prior["weights"], dp_prior_distances=dp_prior["distances"],
        theta_grid=family.theta_grid, theta_prior_mass=family.prior_mass,
        theta_wow_distances=family.wow_distances, centres=cfg.bayes_centres,
        centre_limit=cfg.centre_limit, epsilons=np.asarray(cfg.bayes_epsilons),
        C_values=C_values, N_values=N_values,
    )
    if cfg.bayes_save_data_bank:
        arrays["data_bank"] = data_bank
    np.savez_compressed(out / "raw_results.npz", **arrays)

    probs, draws = fits["param_posterior_probs"], fits["dp_distance_draws"]
    r, c, n, t = _flat_index(probs.shape)
    pd.DataFrame({
        "repetition": r, "C": C_values[c], "n_per_class": N_values[n],
        "theta": family.theta_grid[t], "wow_distance": family.wow_distances[t],
        "prior_probability": family.prior_mass[t], "posterior_probability": probs.ravel(),
    }).to_csv(out / "parametric_posterior_grid.csv", index=False)

    r, c, n, d = _flat_index(draws.shape)
    pd.DataFrame({
        "repetition": r, "C": C_values[c], "n_per_class": N_values[n],
        "posterior_draw": d, "wow_distance": draws.ravel(),
    }).to_csv(out / "dp_posterior_draws.csv", index=False)

    pd.DataFrame({"prior_draw": np.arange(len(dp_prior["distances"])),
                  "wow_distance": dp_prior["distances"]}).to_csv(out / "dp_prior_distances.csv", index=False)

    extra = {"uniform_grid_floor_wow": family.grid_floor,
             "dp_base_weights": dp_base_weights(cfg).tolist()}
    cfg_json = cfg.to_dict()
    cfg_json["derived"] = extra
    (out / "config.json").write_text(json.dumps(cfg_json, indent=2))
    print("Saved files:")
    for p in sorted(out.iterdir()):
        if p.is_file():
            print(f"  {p.name}: {p.stat().st_size:,} bytes")
    return out


def load_bayes_results(directory, with_data: bool = False) -> dict:
    """Load the raw archive (the large `data_bank` only if `with_data`)."""
    with np.load(Path(directory) / "raw_results.npz") as z:
        return {k: z[k] for k in z.files if k != "data_bank" or with_data}


def pivot_repeats(frame: pd.DataFrame, x_col: str, experiment: Optional[str] = None,
                  value_col: str = "distance"):
    """Long-format -> (x values, array of shape (len(x), repetitions))."""
    sel = frame if experiment is None else frame[frame["experiment"] == experiment]
    if sel.empty:
        raise ValueError(f"No rows for experiment {experiment!r}.")
    grid = sel.groupby([x_col, "repetition"])[value_col].first().unstack("repetition")
    if grid.isna().any().any():
        raise ValueError(f"Missing x/repetition observations for {experiment or 'experiment'}.")
    return grid.index.to_numpy(), grid.to_numpy(dtype=float)


def summarise_fixed_c(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / count / 95% CI of the fixed-C distances by n."""
    s = (df[df["experiment"] == "fixed_c"].groupby("n_per_class")["distance"]
         .agg(["mean", "std", "count"]))
    s["ci_95"] = 1.96 * s["std"] / np.sqrt(s["count"])
    return s


# --------------------------------------------------------------------------- #
# 8. Plotting
# --------------------------------------------------------------------------- #
def plot_fixed_c(df: pd.DataFrame, individual: bool = False):
    """Fixed-C convergence: mean + CI band, or individual repetitions."""
    xs, vals = pivot_repeats(df, "n_per_class", "fixed_c")
    mean, ci = mean_ci(vals)
    fig, ax = plt.subplots(figsize=(20, 6), constrained_layout=True)
    if individual:
        ax.plot(xs, vals, linewidth=2, alpha=0.2, color=COLORS[2])
    else:
        ax.plot(xs, mean, linewidth=2, label=r"$W_{W_2}(\widehat P_{C,n}, P_C)$")
    ax.fill_between(xs, np.maximum(mean - ci, 0.0), mean + ci, alpha=0.18)
    ax.set_xlabel("Observed per class samples, $n$", fontsize=20)
    ax.set_ylabel(r"$W_{W_2}$ to true class distribution $P_C$", fontsize=20)
    ax.set_title("Known class distribution: empirical convergence at the WoW level",
                 fontsize=25)
    ax.grid(True, which="both", alpha=0.1)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    return fig


def growing_c_arrays(df: pd.DataFrame) -> dict:
    """Reconstruct x-grids and repetition arrays for the growing-C experiments."""
    C_joint, joint = pivot_repeats(df, "C", "joint_growth")
    C_fixed, fixed = pivot_repeats(df, "C", "fixed_n_control")
    C_cent, cent = pivot_repeats(df, "C", "centre_only")
    n_fixed = df.loc[df["experiment"] == "fixed_n_control", "n_per_class"].dropna().unique()
    if len(n_fixed) > 1:
        raise ValueError("Multiple fixed-n values found in the saved control data.")
    n_joint = (df[df["experiment"] == "joint_growth"].groupby("C")["n_per_class"].first()
               .reindex(C_joint).astype(int).to_numpy())
    return {"C_joint": C_joint, "joint": joint, "C_fixed": C_fixed, "fixed": fixed,
            "C_centre": C_cent, "centre": cent, "n_joint": n_joint,
            "n_fixed": int(n_fixed[0]) if len(n_fixed) else None}


def plot_growing_c(df: pd.DataFrame, individual: bool = False):
    """Growing-C convergence: joint growth vs fixed-n control."""
    a = growing_c_arrays(df)
    jm, jc = mean_ci(a["joint"])
    fm, fc = mean_ci(a["fixed"])
    fig, ax = plt.subplots(figsize=(20, 6), constrained_layout=True)
    lab_j, lab_f = r"joint growth: $n=n(C)$", fr"$C$ growth at fixed $n={a['n_fixed']}$"
    if individual:
        ax.plot(a["C_joint"], a["joint"], linewidth=2, color=COLORS[0], alpha=0.2)
        ax.plot(a["C_fixed"], a["fixed"], linewidth=2, color=COLORS[2], alpha=0.2)
        ax.plot([], [], color=COLORS[0], label=lab_j)
        ax.plot([], [], color=COLORS[2], label=lab_f)
    else:
        ax.plot(a["C_joint"], jm, marker="o", linewidth=2, markersize=4, label=lab_j)
        ax.fill_between(a["C_joint"], np.maximum(jm - jc, 0), jm + jc, alpha=0.18)
        ax.plot(a["C_fixed"], fm, marker="s", linewidth=2, markersize=4, linestyle="--",
                label=lab_f, color=COLORS[2])
        ax.fill_between(a["C_fixed"], np.maximum(fm - fc, 0), fm + fc, alpha=0.12, color=COLORS[2])
    ax.set_ylabel(r"$W_{W_2}(\widehat P_{C,n},P)$", fontsize = 20)
    ax.set_title("WoW distance to the true class population", fontsize=25)
    ax.legend(frameon=False, fontsize=18)
    ax.set_xticks(a["C_joint"], labels=[str(c) for c in a["C_joint"]])
    ax.set_xlabel("Number of sampled class laws, $C$", fontsize=20)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(True, which="both", alpha=0.25)
    return fig


def plot_mnist(df: pd.DataFrame, individual: bool = False, variant: Optional[str] = None,
               band_scale: float = 2.0):
    """MNIST-family train-to-reference discrepancy (band = band_scale x CI half-width)."""
    if variant is None and "variant" in df.columns:
        variant = ", ".join(map(str, df["variant"].dropna().unique())) or "MNIST"
    variant = variant or "MNIST"
    xs, vals = pivot_repeats(df, "n_per_class", "train_to_reference")
    mean, ci = mean_ci(vals)
    fig, ax = plt.subplots(figsize=(20, 6), constrained_layout=True)
    if individual:
        ax.plot(xs, vals, linewidth=2, color=COLORS[2], alpha=0.4)
        ax.set_ylabel(r"Outer $W_{W_2}$ with inner sliced-$W_2$ costs", fontsize=16)
        ax.set_title(f"{variant}: empirical train", fontsize=20)
    else:
        ax.plot(xs, mean, linewidth=2)
        ax.fill_between(xs, np.maximum(mean - band_scale * ci, 0), mean + band_scale * ci, alpha=0.25)
        ax.set_ylabel(r"$W_{W_2}$ of representation to true classes", fontsize=20)
        ax.set_title(f"{variant}: WoW distance of hierarchical empirical representation "
                     "to true class distribution", fontsize=20)
    ax.set_xlabel("Training images per class, $n$", fontsize=20)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(True, which="both", alpha=0.25)
    return fig


def _eps_index(res, epsilon):
    idx = np.flatnonzero(np.isclose(np.asarray(res["epsilons"]), epsilon))
    if len(idx) == 0:
        raise ValueError(f"epsilon={epsilon} not in saved epsilons {res['epsilons'].tolist()}")
    return int(idx[0])


def plot_bayes_mass_by_n(
    res: dict, summ: dict, model: str, epsilon: float, ax=None
):
  """Posterior mass of the WoW epsilon-ball vs C, one curve per n. model: 'param' | 'dp'."""
  mass = summ[f"{model}_mass"]
  ei = _eps_index(res, epsilon)
  names = {"param": "Parametric hyperprior", "dp": "Finite-DP hyperprior"}

  created_fig = False
  if ax is None:
    fig, ax = plt.subplots(figsize=(10, 4))
    created_fig = True
  else:
    fig = ax.figure

  for ni, n in enumerate(res["N_values"]):
    mean, ci = mean_ci(mass[:, :, ni, ei], axis=0)
    ax.plot(res["C_values"], mean, label=f"n={n}")
    ax.fill_between(
        res["C_values"],
        np.maximum(0, mean - ci),
        np.minimum(1, mean + ci),
        alpha=0.12,
    )

  ax.set_xlabel("Number of sampled classes C")
  ax.set_ylabel(rf"Posterior mass: $W_{{W_2}}(P,P^*)<{epsilon:.2f}$")
  ax.set_title(f"{names[model]}: posterior concentration near the true P")
  ax.grid(alpha=0.2)

  if created_fig:
    fig.tight_layout()
  return fig


def plot_bayes_mass_comparison(res: dict, summ: dict, epsilon: float, n_index: int = -1):
    """Both hyperpriors on one axis (error bars = 95% CI across repetitions)."""
    ei = _eps_index(res, epsilon)
    pm, pc = mean_ci(summ["param_mass"][:, :, n_index, ei], axis=0)
    dm, dc = mean_ci(summ["dp_mass"][:, :, n_index, ei], axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(res["C_values"], pm, yerr=pc, marker="o", capsize=3, label="Parametric hyperprior")
    ax.errorbar(res["C_values"], dm, yerr=dc, marker="s", capsize=3, label="Finite-DP hyperprior")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Number of classes C")
    ax.set_ylabel(rf"Posterior mass in $B_{{{epsilon:g}}}(P^*)$")
    ax.set_title(f"Posterior mass, n={res['N_values'][n_index]} per class")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_bayes_distance(res: dict, summ: dict, n_index: int = -1):
    """Posterior mean WoW distance to the truth vs C for both hyperpriors."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
    for ax, key, title in [(axes[0], "param_mean_distance", "Parametric hyperprior"),
                           (axes[1], "dp_mean_distance", "Finite-DP hyperprior")]:
        mean, ci = mean_ci(summ[key][:, :, n_index], axis=0)
        ax.plot(res["C_values"], mean)
        ax.fill_between(res["C_values"], np.maximum(0, mean - ci), mean + ci, alpha=0.2)
        ax.set_xlabel("Number of classes C")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"Posterior mean $W_{W_2}(P,P^*)$")
    fig.suptitle(f"Posterior distance to truth with n={res['N_values'][n_index]} per class")
    fig.tight_layout()
    return fig


def plot_theta_posterior(res: dict):
    """Parametric posterior density over theta at the smallest and largest settings."""
    theta = res["theta_grid"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for ci, ni in [(0, 0), (-1, -1)]:
        post = res["param_posterior_probs"][:, ci, ni, :].mean(axis=0)
        ax.plot(theta, post / np.gradient(theta),
                label=f"C={res['C_values'][ci]}, n={res['N_values'][ni]}")
    ax.axvline(0.0, color="black", linestyle="--", label=r"truth $\theta^*=0$")
    ax.set_xlabel(r"Population parameter $\theta$")
    ax.set_ylabel("Posterior density (grid approximation)")
    ax.set_title("Parametric hyperposterior concentration")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_bayes_mass_all_epsilons(
    res: dict,
    summ: dict,
    model: str = "both",
    n_index: int = -1,
    show_ci: bool = False,
):
    """Posterior mass vs C, one curve per epsilon.

    model: 'param', 'dp', or 'both'.
    show_ci: display 95% normal-approximation confidence bands.
    """
    if model not in {"param", "dp", "both"}:
        raise ValueError("model must be 'param', 'dp', or 'both'")

    names = {
        "param": "Parametric hyperprior",
        "dp": "Finite-DP hyperprior",
    }

    models = ["param", "dp"] if model == "both" else [model]

    if model == "both":
        fig, axes = plt.subplots(
            1, 2, figsize=(20, 6), sharey=True
        )
    else:
        fig, ax = plt.subplots(figsize=(20, 6))
        axes = [ax]

    epsilons = np.asarray(res["epsilons"])
    colours = plt.cm.viridis(
        np.linspace(0, 0.5, len(epsilons))
    )

    for ax, current_model in zip(axes, models):
        mass = summ[f"{current_model}_mass"]

        for ei, (epsilon, colour) in enumerate(
            zip(epsilons, colours)
        ):
            mean, ci = mean_ci(
                mass[:, :, n_index, ei], axis=0
            )

            ax.plot(
                res["C_values"],
                mean,
                color=colour,
                linewidth=2,
                label=fr"$\epsilon={epsilon:g}$",
            )

            if show_ci:
                ax.fill_between(
                    res["C_values"],
                    np.maximum(0, mean - ci),
                    np.minimum(1, mean + ci),
                    color=colour,
                    alpha=0.10,
                )

        ax.set_title(names[current_model], fontsize = 18)
        ax.set_xlabel("Number of sampled classes C", fontsize = 18)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(title=fr"$\varepsilon$ ball around the truth", frameon=False)

    axes[0].set_ylabel(fr"Posterior mass in WoW $\varepsilon$ ball", fontsize = 20)

    fig.suptitle(
        fr"Posterior mass concentrated on WoW $\varepsilon$ ball", fontsize = 25
    )
    fig.tight_layout()

    return fig