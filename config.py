"""Shared default parameters for the WoW convergence / Bayesian hyperprior experiments.

Every notebook does::

    OVERRIDES = {}                      # e.g. dict(meta_repeats=4, results_root="results_test")
    cfg = make_config(**OVERRIDES)

so the defaults live here once, and any notebook can deviate from them locally
without editing this file.  Unknown field names raise immediately.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _inclusive_range(start, stop, step):
    """Integer grid start, start+step, ..., <= stop (stop inclusive)."""
    return np.arange(start, stop + 1, step, dtype=int)


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ global
    seed: int = 13
    results_root: str = "results"

    # ------------------------------------------------- ring model (all notebooks)
    ring_radius: float = 0.7
    ring_radial_jitter: float = 0.045
    ring_support_size: int = 2048          # must be even (antipodal jitter pairing)
    centre_limit: float = 2.0              # P = Law(mu_A), A ~ Uniform[-L, L]

    # ---------------------------- Experiment 1 (convergence): fixed class count C
    # Three rings at x = -L, 0, +L unless `fixed_c_centres_x` is given explicitly.
    fixed_c_centres_x: Optional[Tuple[float, ...]] = None
    fixed_c_n_start: int = 10
    fixed_c_n_stop: int = 1000             # inclusive
    fixed_c_n_step: int = 10
    fixed_c_repeats: int = 16

    # ----------------------- Experiment 2 (convergence): growing C (and growing n)
    meta_c_start: int = 5
    meta_c_stop: int = 100                 # inclusive
    meta_c_step: int = 5
    # n(C) grows geometrically from `meta_n_joint_min` to `meta_n_joint_max`.
    meta_n_joint_min: int = 8
    meta_n_joint_max: int = 2048
    meta_n_fixed: int = 10                 # control: C grows, n stays here
    meta_repeats: int = 16

    # ----------------------------------------------- MNIST-family diagnostic
    mnist_variant: str = "MNIST"           # MNIST | FashionMNIST | KMNIST | USPS
    mnist_image_size: int = 12
    mnist_n_start: int = 4
    mnist_n_stop: int = 255                # inclusive
    mnist_repeats: int = 32
    mnist_reference_size: int = 4096       # per class, in each of two disjoint banks
    mnist_sw_projections: int = 64
    mnist_quantile_points: int = 128
    mnist_data_root: str = "./data"

    # ------------------------------------------------ Bayesian hyperprior study
    bayes_n_centres: int = 200
    bayes_c_start: int = 50
    bayes_c_stop: int = 5000               # inclusive
    bayes_c_step: int = 50
    bayes_n_values: Tuple[int, ...] = (1000,)   # non-decreasing
    bayes_repeats: int = 16
    bayes_epsilons: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.2, 0.25)

    # parametric (exponential-tilt) hyperprior
    bayes_theta_min: float = -6.0
    bayes_theta_max: float = 6.0
    bayes_theta_points: int = 501
    bayes_theta_prior_mean: float = -0.8
    bayes_theta_prior_sd: float = 1.8

    # finite-support Dirichlet-process hyperprior
    bayes_dp_alpha: float = 0.5
    bayes_dp_base_sd: float = 1.0          # base weights H_j ∝ exp(-a_j^2 / (2 sd^2))
    bayes_dp_burn_in: int = 20
    bayes_dp_draws: int = 500
    bayes_dp_thin: int = 1
    bayes_dp_prior_draws: int = 500

    # numerics / bookkeeping
    bayes_loglik_chunk: int = 16
    bayes_class_chunk: int = 32
    bayes_checkpoint_every: int = 10
    bayes_save_data_bank: bool = True      # the raw bank is ~1.3 GB at the default design

    # ------------------------------------------------------------ derived values
    @property
    def results_dir(self) -> Path:
        return Path(self.results_root)

    @property
    def fixed_c_centres(self) -> np.ndarray:
        xs = self.fixed_c_centres_x
        if xs is None:
            xs = (-self.centre_limit, 0.0, self.centre_limit)
        return np.column_stack([np.asarray(xs, float), np.zeros(len(xs))])

    @property
    def fixed_c_sample_sizes(self) -> np.ndarray:
        return _inclusive_range(self.fixed_c_n_start, self.fixed_c_n_stop, self.fixed_c_n_step)

    @property
    def meta_c_values(self) -> np.ndarray:
        return _inclusive_range(self.meta_c_start, self.meta_c_stop, self.meta_c_step)

    @property
    def meta_n_joint(self) -> np.ndarray:
        """n(C): geometric from meta_n_joint_min to meta_n_joint_max over the C grid."""
        c = self.meta_c_values
        return np.round(np.geomspace(self.meta_n_joint_min, self.meta_n_joint_max, len(c))).astype(int)

    @property
    def mnist_sample_sizes(self) -> np.ndarray:
        return _inclusive_range(self.mnist_n_start, self.mnist_n_stop, 1)

    @property
    def bayes_centres(self) -> np.ndarray:
        """Midpoint quadrature nodes on [-L, L]."""
        L, K = self.centre_limit, self.bayes_n_centres
        width = 2.0 * L / K
        return -L + (np.arange(K) + 0.5) * width

    @property
    def bayes_c_values(self) -> np.ndarray:
        return _inclusive_range(self.bayes_c_start, self.bayes_c_stop, self.bayes_c_step)

    @property
    def bayes_theta_grid(self) -> np.ndarray:
        return np.linspace(self.bayes_theta_min, self.bayes_theta_max, self.bayes_theta_points)

    # ------------------------------------------------------------------- helpers
    def override(self, **changes) -> "Config":
        valid = {f.name for f in fields(self)}
        unknown = sorted(set(changes) - valid)
        if unknown:
            raise TypeError(f"Unknown config field(s): {unknown}. Valid fields: {sorted(valid)}")
        return replace(self, **changes)

    def diff_from_defaults(self) -> dict:
        default = Config()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(default, f.name)}

    def to_dict(self) -> dict:
        out = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}


DEFAULTS = Config()


def make_config(**overrides) -> Config:
    """Return the default configuration with `overrides` applied (validated)."""
    cfg = DEFAULTS.override(**overrides)
    if cfg.ring_support_size % 2:
        raise ValueError("ring_support_size must be even for antipodal pairing.")
    if list(cfg.bayes_n_values) != sorted(cfg.bayes_n_values):
        raise ValueError("bayes_n_values must be non-decreasing.")
    return cfg


def describe(cfg: Config) -> None:
    changed = cfg.diff_from_defaults()
    if not changed:
        print("Using config.py defaults.")
    else:
        print("Config overrides relative to config.py defaults:")
        for k, v in changed.items():
            print(f"  {k} = {v!r}")
