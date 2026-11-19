# L2-over-Wasserstein
Code reproducing the experiments from L2 over Wasserstein random variables: Statistical analysis for optimal transport
=======
```
config.py                         defaults (one dataclass, `make_config(**OVERRIDES)`)
utils.py                          all shared code (ring, transport, experiments, Bayes, I/O, plots)
01_convergence_experiments.ipynb  fixed-C + growing-C  (old class_distribution_convergence_wow_optimized, minus MNIST)
02_mnist_diagnostic.ipynb         MNIST-family diagnostic (only notebook needing torch)
03_bayesian_hyperprior.ipynb      old bayesian_hyperprior_aligned_optimized
04_plots.ipynb                    all figures from saved results (old ..._plots + Bayesian plots)
```

Each notebook starts with `OVERRIDES = {}`; put any `Config` field there to deviate from `config.py` for that notebook only.

## Results layout (all under `cfg.results_root`, default `results/`)
| file | content |
|---|---|
| `convergence/convergence.csv` | long format `experiment, C, n_per_class, repetition, distance, seed`; experiments `fixed_c`, `joint_growth`, `fixed_n_control`, `centre_only` |
| `mnist/<variant>.csv` | same columns + `variant`; experiments `train_to_reference`, `reference_split` |
| `bayes/raw_results.npz`, `parametric_posterior_grid.csv`, `dp_posterior_draws.csv`, `dp_prior_distances.csv` | Bayesian study |
| `*/config.json` | the exact config used |
| `bayes/checkpoints/` | resumable; invalidated automatically when any relevant setting changes |
