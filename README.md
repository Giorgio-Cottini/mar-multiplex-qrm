# Dynamic Multiplex Financial Networks via Matrix Autoregression

This repository estimates time-varying spillover networks among a panel of Dow
Jones constituents by combining a Matrix-valued Autoregressive (MAR) model with a
Multiplex spatial-network model. Adjacency matrices are extracted from rolling-window
MAR fits (on returns, on log realized variances, or on rolling Quantile Dependence
Matrices), then fed as layers into a Multiplex network whose composite weight matrix
`W*` and spatial-autoregression parameters are estimated by concentrated maximum
likelihood against Fama–French 5-factor returns. Outputs are rolling-window parameter
series and animated network graphs at several rebalancing frequencies (daily, weekly,
monthly, bimonthly). The data are end-of-day prices and 1-minute realized variances
(Capire, March 2025 snapshot) plus daily FF5 factors.

## Methodology

The pipeline implements two model families.

**MAR — Matrix-valued Autoregression** (`src/models.py`, class `MAR`). Direct
implementation of the one-lag MAR of Chen, Xiao & Yang (2020): `X_t ≈ A · X_{t-1} · B`
where `X_t` is an `m × n` matrix observation. Starting values come from the Projection
(PROJ) estimator — an unrestricted VAR(1) coefficient matrix `Φ̂` is rearranged and an
SVD recovers `A` and `B`. Parameters `(A, B, Σ_c, Σ_r)` are then refined by an
alternating MLE update, with Frobenius-norm normalization preserving the Kronecker
structure `W = B ⊗ A`, `Σ_W = Σ_c ⊗ Σ_r`. Convergence is monitored by relative
log-likelihood improvement with a patience-based early stop, and a full diagnostic
suite reports spectral radii, eigenvalue gaps/ratios, condition numbers, and covariance
anisotropies.

**Multiplex network** (`src/models.py`, class `Multiplex`). Based on the spatial
multiplex specification of Bonaccolto, Caporin & Paterlini (2019). Given `d` layer
adjacency matrices `W_j` (diagonal zeroed, row- or spectral-normalized), it forms a
composite network `W* = Σ_j δ_j W_j`, a spatial filter `A = I − diag(ρ) · W*`, and
estimates `(δ, ρ)` by concentrated MLE — filtered returns `A·R` are regressed on the
factors `F` (no intercept) and the profile log-likelihood
`ℓ = T·log|det A| − (NT/2)·log σ̂²` is maximized via SLSQP under
`Σ δ_j = 1`, `δ_j ∈ [0,1]`, `ρ_i ∈ [−0.99, 0.99]`.

**Quantile Dependence Matrices** (`src/trainer.py`, `quantile_dependence_matrix`,
`get_qdm_rolling`). Tail co-movement matrices built from the correlation of
lower-quantile exceedance indicators, computed on rolling windows to produce a
matrix-valued time series that the MAR model then tracks.

The end-to-end estimators in `src/trainer.py` combine these:

- `fit_returns_MAR` — rolling MAR fit on a single panel, returns a diagnostics DataFrame.
- `fit_MAR_Multiplex` — MAR on returns and on log-volatilities; the two Kronecker
  matrices `W` become the two layers of a Multiplex network.
- `fit_qdm_MAR` — MAR fit on a rolling QDM tensor.
- `fit_qdm_Multiplex_MAR` — MAR on rolling QDMs; the per-window `A` and `Bᵀ` become a
  2-layer Multiplex network (the main pipeline used in `main.ipynb`).
- `fit_qdm_Multiplex_MAR_split` — 4-layer variant splitting `A` and `B` into positive
  and negative parts.
- `fit_qdm_MultipMAR_ret_vol` — 4-layer variant whose layers are the returns and
  volatility MAR matrices `[A_ret, Bᵀ_ret, A_vol, Bᵀ_vol]`.

Plotting and diagnostics live in `src/plotter.py`: spillover indices, delta-weight
paths, network in/out-strength with EWMA bands, and spectral systemic-risk metrics
(cumulative spectral power, von Neumann entropy, non-normality index, IPR, Katz
centrality), plus `animate_results`, which renders the time-varying `W*` as a directed
graph animation.

## Repository layout

```txt
homework3/
├── Dataset/
│   ├── raw/                     raw CSVs (Capire March 2025, FF5 daily) — not shipped
│   └── dataset.xlsx             processed workbook (EoD_Prices, Realized_Variances, FF5) — not shipped
├── articles/                    reference PDFs — not shipped (copyright)
├── results/<frequency>/         per-frequency outputs
│   ├── *.mp4                     network animations (shipped)
│   └── *.npz                     parameter archives (not shipped — see Data availability)
├── src/
│   ├── loader.py                data loading, asset-grid layout, QDM helpers, screening
│   ├── models.py                MAR and Multiplex model classes
│   ├── trainer.py               rolling-window estimators and QDM construction
│   └── plotter.py               diagnostics, systemic metrics, animation
├── preprocess_dataset.ipynb     builds Dataset/dataset.xlsx from raw CSVs
├── main.ipynb                   full estimation pipeline across frequencies
├── convergence_test.ipynb       MAR convergence experiments
└── update.py                    directory-tree printer (developer utility)
```

## Requirements

Python 3.12. Install dependencies:

```bash
pip install numpy pandas scipy matplotlib networkx tqdm statsmodels openpyxl
```

Saving `.mp4` animations requires `ffmpeg` on the system `PATH`.

## How to run

The dataset workbook is built from the raw CSVs and is consumed by the rest of the
pipeline. Run in order:

1. **Build the dataset** — run `preprocess_dataset.ipynb`. The first cell merges the
   Capire price and realized-variance CSVs into `Dataset/dataset.xlsx` (sheets
   `EoD_Prices`, `Realized_Variances`); the second cell appends the cleaned FF5 daily
   factors as sheet `FF5`. Requires `Dataset/raw/` to be present.

2. **Estimate the models** — run `main.ipynb`. It loads the workbook via
   `loader.load_formatted_data`, screens the panel down to 15 assets
   (`loader.screen_assets`), builds rolling QDMs, and calls the `fit_qdm_Multiplex_MAR`
   family at four step sizes: `step=1` (daily), `5` (weekly), `22` (monthly), `44`
   (bimonthly). Each run writes an `.npz` to `results/<frequency>/` and, for the QDM
   pipeline, an `.mp4` animation. Part 3 of the notebook reloads each `.npz` and renders
   the diagnostic and systemic-metric figures.

3. **(Optional) Convergence diagnostics** — `convergence_test.ipynb` fits a single MAR
   to convergence and plots the log-likelihood and spectral-radius paths.

The window length is fixed at `CONTEXT_LENGTH = 252` (one trading year) throughout.

## Output

Each estimator writes a compressed `.npz` whose arrays are indexed by rolling-window
end date. For the 2-layer QDM pipeline the keys are:

| Key | Shape | Contents |
|-----|-------|----------|
| `dates` | `(T,)` | window end dates |
| `A`, `B` | `(T, N, N)` | per-window MAR matrices |
| `W1`, `W2` | `(T, N, N)` | normalized Multiplex layers |
| `W_star` | `(T, N, N)` | composite network `W*` |
| `delta_hat` | `(T, d)` | layer weights |
| `rho_hat` | `(T, N)` | spatial-autoregression coefficients |
| `beta_hat` | `(T, K, N)` | factor loadings |
| `sigma_sq` | `(T,)` | residual variance |

(`N = 15` screened assets, `K = 5` factors. The 4-layer variants add `W3`, `W4` and,
for the returns/volatility model, `A_r`, `B_r`, `A_v`, `B_v`.) The `.mp4` files in each
`results/<frequency>/` folder are the corresponding `W*` network animations.

## Results

The figures below are produced inside `main.ipynb` (Part 1 and Part 3) and the network
animations are the `.mp4` files in `results/<frequency>/`. The findings are descriptive
and exploratory — there is no out-of-sample test, no backtest against dated systemic
events, and no significance bands on the parameter paths. They should be read as such.

**MAR fit.** The rolling log-likelihood drops sharply during 2008–09 and 2020, tracking
market stress. Spectral radii `ρ_A`, `ρ_B` sit below the stationarity bound in most
windows, but `ρ_A` breaches 1.0 in a number of windows — the MAR fit is not stationary
everywhere. Most windows converge in roughly ten EM iterations.

**Spillovers.** Total spillover indices for `A`, `B`, and the composite `W*` spike at
the 2008, 2011, and 2020 episodes across frequencies. The net-cancellation series of
`W*` is large and positive, i.e. the composite network is substantially smaller than the
weighted sum of its layers — positive and negative spillovers offset heavily. The
non-normality index spikes at crises, consistent with transient amplification under
non-normal transition dynamics.

**Layer weights.** The estimated layer weights `δ_j` are effectively bang-bang: they
snap to 0 or 1 and switch regimes rather than settling at interior values (the animation
frames confirm this — e.g. a week with `δ_A = 0.00, δ_B = 1.00`). This follows from the
`Σ δ_j = 1` constraint combined with near-collinear layers, which pushes the optimizer to
corner solutions. The layer weights are therefore weakly identified and should not be
read as a stable decomposition of which layer dominates.

**Animations.** Each `results/<frequency>/multiplex_animation.mp4` renders the
time-varying composite network `W*` as a directed graph (red = positive, blue = negative
edges, shown above the 95th within-frame percentile). Node positions are fixed on a
circle; CSCO and CVX recur as hubs. The four files differ only in rebalancing step
(daily/weekly/monthly/bimonthly) and frame rate; the daily series is the noisiest, the
coarser steps smooth the same regime pattern.

### Known numerical caveats

- **Tail covariance blow-up.** In the MAR returns diagnostics, the covariance anisotropy
  and noise-scale ratio grow steeply over 2023–2025, indicating `Σ_r` becoming
  ill-conditioned near the end of the sample. This is likely numerical rather than
  economic and is not corrected.
- **Katz centrality scaling.** Katz centrality is computed with a fixed damping
  `α = 0.9`. When the spectral radius of `W*` approaches `1/α`, the matrix
  `(I − α·W*ᵀ)` becomes near-singular and the centrality spikes to very large values
  (~10⁴ in some windows). These spikes are a numerical artifact of the fixed `α`, not a
  systemic-risk signal; an `α` scaled to the per-window spectral radius, or a normalized
  centrality, would remove them.

## Data availability

The raw data (`Dataset/raw/`) and the processed workbook (`Dataset/dataset.xlsx`) are
**not** included: the Capire price and realized-variance series are proprietary and not
redistributable here. The Fama–French factors are publicly available from the Kenneth
French Data Library. The reference PDFs under `articles/` are **not** included for
copyright reasons.

The `.npz` parameter archives are **not** shipped — the daily-frequency files in
particular are large. Only the `.mp4` network animations are committed, as a
lightweight visual record of the estimated dynamics. To regenerate the `.npz` files,
supply the dataset and run `main.ipynb`.
