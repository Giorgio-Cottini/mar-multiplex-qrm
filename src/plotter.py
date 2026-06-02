import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple

# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def visualize_full_diagnostics(
    results_df: pd.DataFrame,
    suptitle: str = "You should've chosen a title, now you're stuck with this one",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Visualize all diagnostics from a full run of fit_MAR with full_diagnostics=True.

    Args:
        results_df: DataFrame returned by fit_MAR with full_diagnostics=True.
    Returns:
        tuple[plt.Figure, np.ndarray]: Figure and axes array.
    """
    AUX_COLOR = "#287271"
    B_COLOR = "#FF7F0E"
    A_COLOR = "#1F77B4"
    W_COLOR = "#2CA02C"
    # Create figure with 8 subplots (4 rows x 2 columns)
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()

    # Use window_end_date as x-axis if available, otherwise use index
    if "window_end_date" in results_df.columns:
        x = results_df["window_end_date"]
        xlabel = "Window End Date"
    else:
        x = results_df.index
        xlabel = "Window Index"

    # Plot 1: Final Log-Likelihood
    axes[0].plot(x, results_df["final_ll"], color=AUX_COLOR, linewidth=1.5)
    axes[0].set_title("Final Log-Likelihood", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Log-Likelihood")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Number of Iterations
    axes[1].plot(
        x,
        results_df["iterations"],
        color=AUX_COLOR,
        linewidth=1.5,
    )
    axes[1].set_title("Number of Iterations", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Iterations")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Spectral Radii (rho_A, rho_B, rho_W)
    axes[2].plot(
        x, results_df["rho_A"], label="ρ_A", linewidth=1.2, alpha=0.8, color=A_COLOR
    )
    axes[2].plot(
        x, results_df["rho_B"], label="ρ_B", linewidth=1.2, alpha=0.8, color=B_COLOR
    )
    axes[2].plot(
        x, results_df["rho_W"], label="ρ_W", linewidth=1.2, alpha=0.8, color=W_COLOR
    )
    axes[2].axhline(y=1.0, color="red", linewidth=0.75, label="Stationarity Bound")
    axes[2].set_title("Spectral Radii", fontsize=12, fontweight="bold")
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel("Spectral Radius")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Eigenvalue Gaps
    axes[3].plot(
        x,
        results_df["eig_gap_A"],
        label="Gap A",
        linewidth=1.2,
        alpha=0.8,
        color=A_COLOR,
    )
    axes[3].plot(
        x,
        results_df["eig_gap_B"],
        label="Gap B",
        linewidth=1.2,
        alpha=0.8,
        color=B_COLOR,
    )
    axes[3].plot(
        x,
        results_df["eig_gap_W"],
        label="Gap W",
        linewidth=1.2,
        alpha=0.8,
        color=W_COLOR,
    )
    axes[3].set_title("Eigenvalue Gaps", fontsize=12, fontweight="bold")
    axes[3].set_xlabel(xlabel)
    axes[3].set_ylabel("Gap (λ₁ - λ₂)")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    # Plot 5: Eigenvalue Ratios
    axes[4].plot(
        x,
        results_df["eig_ratio_A"],
        label="Ratio A",
        linewidth=1.2,
        alpha=0.8,
        color=A_COLOR,
    )
    axes[4].plot(
        x,
        results_df["eig_ratio_B"],
        label="Ratio B",
        linewidth=1.2,
        alpha=0.8,
        color=B_COLOR,
    )
    axes[4].plot(
        x,
        results_df["eig_ratio_W"],
        label="Ratio W",
        linewidth=1.2,
        alpha=0.8,
        color=W_COLOR,
    )
    axes[4].set_title("Eigenvalue Ratios", fontsize=12, fontweight="bold")
    axes[4].set_xlabel(xlabel)
    axes[4].set_ylabel("Ratio (λ₁ / λ₂)")
    axes[4].legend(loc="best")
    axes[4].grid(True, alpha=0.3)

    # Plot 6: Noise Scale Ratio
    axes[5].plot(
        x, results_df["noise_scale_ratio"], color=AUX_COLOR, linewidth=1.5, alpha=0.8
    )
    axes[5].set_title("Noise Scale Ratio", fontsize=12, fontweight="bold")
    axes[5].set_xlabel(xlabel)
    axes[5].set_ylabel("||Σ_c|| / ||Σ_r||")
    axes[5].grid(True, alpha=0.3)

    # Plot 7: Log Determinants of Covariances
    axes[6].plot(
        x,
        results_df["log_det_Sigma_c"],
        label="log|Σ_c|",
        linewidth=1.2,
        alpha=0.8,
        color=A_COLOR,
    )
    axes[6].plot(
        x,
        results_df["log_det_Sigma_r"],
        label="log|Σ_r|",
        linewidth=1.2,
        alpha=0.8,
        color=B_COLOR,
    )
    axes[6].plot(
        x,
        results_df["log_det_Sigma_W"],
        label="log|Σ_W|",
        linewidth=1.2,
        alpha=0.8,
        color=W_COLOR,
    )
    axes[6].set_title("Log Determinants of Covariances", fontsize=12, fontweight="bold")
    axes[6].set_xlabel(xlabel)
    axes[6].set_ylabel("Log Determinant")
    axes[6].legend(loc="best")
    axes[6].grid(True, alpha=0.3)

    # Plot 8: Anisotropies
    axes[7].plot(
        x,
        results_df["Sigma_c_anisotropy"],
        label="Σ_c",
        linewidth=1.2,
        alpha=0.8,
        color=A_COLOR,
    )
    axes[7].plot(
        x,
        results_df["Sigma_r_anisotropy"],
        label="Σ_r",
        linewidth=1.2,
        alpha=0.8,
        color=B_COLOR,
    )
    axes[7].plot(
        x,
        results_df["Sigma_W_anisotropy"],
        label="Σ_W",
        linewidth=1.2,
        alpha=0.8,
        color=W_COLOR,
    )
    axes[7].set_title("Covariance Anisotropies", fontsize=12, fontweight="bold")
    axes[7].set_xlabel(xlabel)
    axes[7].set_ylabel("Anisotropy (λ_max / λ_min)")
    axes[7].legend(loc="best")
    axes[7].grid(True, alpha=0.3)

    # Configure x-axis ticks for all subplots
    # Set yearly ticks and rotate labels 45 degrees
    if "window_end_date" in results_df.columns:
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(axis="x", rotation=45)

    # Add super title
    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=0.995)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    plt.show()

    return fig, axes


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def animate_results(
    multiplex_params: Union[str, Dict[str, Any]],
    *,
    asset_names: Optional[List[str]] = None,
    returns: Optional[pd.DataFrame] = None,
    layout: str = "circular",
    edge_percentile: float = 95.0,
    global_scale_percentile: float = 99.0,
    width_factor: float = 4.0,
    interval: int = 200,
    save_path: Optional[str] = None,
    fps: int = 5,
    dpi: int = 150,
):
    """
    Animate the fitted composite network W* over time using outputs from
    `fit_qdm_Multiplex_MAR` (saved payload or already-loaded dict).

    Expected payload keys (from `fit_qdm_Multiplex_MAR`):
    - dates: (T,) datetime64 array
    - A: (T, N, N) ndarray
    - B: (T, N, N) ndarray
    - delta_hat: (T, 2) ndarray (for num_layers=2)

    Args:
        multiplex_params:
            - path to `.npz` produced by `fit_qdm_Multiplex_MAR`, OR
            - a dict-like payload already in memory (same keys as the `.npz`).
        asset_names: list of node names (length N). If None, uses returns.columns.
        returns: optional DataFrame only used to infer asset_names.
        layout: "circular" or "spring"
        edge_percentile: show edges with |w| above this per-frame percentile.
        global_scale_percentile: percentile over all |w| across time for width scaling.
        width_factor: factor for edge width scaling.
        interval: milliseconds between frames.
        save_path: if provided (e.g. "out.mp4"), saves the animation there.
        fps: frames per second for saving.
        dpi: resolution for saving.

    Returns:
        matplotlib.animation.FuncAnimation
    """

    # Define parameters
    figsize = (9, 9)
    node_size = 900
    node_color = "lightgrey"
    arrowsize = 10
    alpha = 0.75
    title_prefix = "Network W* — week"
    # Local imports to keep the module lightweight when not animating.
    from pathlib import Path

    import networkx as nx
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import writers

    def _load_payload(x: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(x, dict):
            return x
        p = Path(x)
        if not p.exists():
            raise FileNotFoundError(f"Could not find params file: {p}")
        if p.suffix.lower() != ".npz":
            raise ValueError(
                "multiplex_params must be a '.npz' file (only supported format)."
            )
        data = np.load(p, allow_pickle=False)
        return {k: data[k] for k in data.files}

    def _preprocess_adj(matrix: np.ndarray) -> np.ndarray:
        """Match `Multiplex.set_adjacency_matrices`: zero diagonal + row-normalize."""
        M = np.asarray(matrix, dtype=float).copy()
        np.fill_diagonal(M, 0.0)
        row_sums = np.sum(M, axis=1, keepdims=True)
        row_sums_safe = np.where(row_sums == 0.0, 1.0, row_sums)
        M = M / row_sums_safe
        M = np.where(row_sums == 0.0, 0.0, M)
        return M

    payload = _load_payload(multiplex_params)
    if (
        "dates" not in payload
        or "A" not in payload
        or "B" not in payload
        or "delta_hat" not in payload
    ):
        raise KeyError("Payload must contain keys: 'dates', 'A', 'B', 'delta_hat'.")

    dates = pd.to_datetime(payload["dates"])
    A_store = np.asarray(payload["A"], dtype=float)
    B_store = np.asarray(payload["B"], dtype=float)
    delta_store = np.asarray(payload["delta_hat"], dtype=float)

    if A_store.ndim != 3 or B_store.ndim != 3:
        raise ValueError(
            f"Expected A and B with shape (T,N,N). Got {A_store.shape=} {B_store.shape=}"
        )
    if A_store.shape != B_store.shape:
        raise ValueError(
            f"A and B must have the same shape. Got {A_store.shape=} {B_store.shape=}"
        )
    T, N, N2 = A_store.shape
    if N != N2:
        raise ValueError(f"Expected square matrices; got {A_store.shape=}")
    if len(dates) != T:
        raise ValueError(f"dates length must match T. Got {len(dates)=} vs {T=}")
    if delta_store.shape[0] != T:
        raise ValueError(
            f"delta_hat must have first dim T. Got {delta_store.shape=} vs {T=}"
        )

    if asset_names is None:
        if returns is None:
            asset_names = [f"asset_{i}" for i in range(N)]
        else:
            asset_names = list(returns.columns)
    if len(asset_names) != N:
        raise ValueError(f"asset_names length must be N={N}. Got {len(asset_names)}.")

    # Fixed node positions
    G0 = nx.DiGraph()
    G0.add_nodes_from(asset_names)
    if layout == "circular":
        pos = nx.circular_layout(G0)
    elif layout == "spring":
        pos = nx.spring_layout(G0, seed=0)
    else:
        raise ValueError("layout must be 'circular' or 'spring'")
    pos_arr = np.asarray(list(pos.values()), dtype=float)
    x_min, x_max = float(pos_arr[:, 0].min()), float(pos_arr[:, 0].max())
    y_min, y_max = float(pos_arr[:, 1].min()), float(pos_arr[:, 1].max())
    pad = 0.1 * max(x_max - x_min, y_max - y_min, 1e-12)
    xlim = (x_min - pad, x_max + pad)
    ylim = (y_min - pad, y_max + pad)

    # Precompute W* and global scale for consistent edge widths
    Wstar_all = np.empty((T, N, N), dtype=float)
    for t in range(T):
        W1 = _preprocess_adj(A_store[t])
        W2 = _preprocess_adj(B_store[t])
        d = delta_store[t]
        if d.shape[0] < 2:
            raise ValueError(
                "delta_hat must have at least 2 entries per time step (num_layers=2)."
            )
        Wstar_all[t] = d[0] * W1 + d[1] * W2
        np.fill_diagonal(Wstar_all[t], 0.0)

    abs_vals = np.abs(Wstar_all).ravel()
    global_scale = np.percentile(abs_vals, global_scale_percentile)
    global_scale = float(max(global_scale, 1e-12))

    fig, ax = plt.subplots(figsize=figsize)

    def update(frame_idx: int):
        ax.clear()
        date = dates[frame_idx]
        W = Wstar_all[frame_idx]

        # Show only strongest links for the frame (exclude diagonal by construction)
        thr = np.percentile(np.abs(W).ravel(), edge_percentile)

        G = nx.DiGraph()
        G.add_nodes_from(asset_names)
        for i in range(N):
            for j in range(N):
                w = W[i, j]
                if abs(w) > thr:
                    # Edge direction matches your baseline: j -> i corresponds to W[i,j]
                    G.add_edge(asset_names[j], asset_names[i], weight=w)

        edges = list(G.edges())
        edge_colors = ["red" if G[u][v]["weight"] > 0 else "blue" for u, v in edges]
        edge_widths = [
            min(abs(G[u][v]["weight"]), global_scale) / global_scale * width_factor
            for u, v in edges
        ]

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_size,
            node_color=node_color,
            ax=ax,
        )
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_colors,
            width=edge_widths,
            arrowsize=arrowsize,
            alpha=alpha,
            ax=ax,
        )

        # Helpful caption: weights for layers (if present)
        d = delta_store[frame_idx]
        d_txt = ""
        if d.shape[0] >= 2:
            d_txt = f"  (δ_A={d[0]:.2f}, δ_B={d[1]:.2f})"

        try:
            date_str = pd.Timestamp(date).date()
        except Exception:
            date_str = str(date)
        ax.set_title(f"{title_prefix} {date_str}{d_txt}", fontsize=14)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    ani = FuncAnimation(fig, update, frames=T, interval=interval)

    if save_path is not None:
        if not str(save_path).lower().endswith(".mp4"):
            raise ValueError("save_path must end with '.mp4'")
        if not writers.is_available("ffmpeg"):
            raise RuntimeError(
                "Saving MP4 requires ffmpeg. Install ffmpeg and ensure it's on PATH, "
                "then restart the kernel."
            )
        ani.save(save_path, writer="ffmpeg", fps=fps, dpi=dpi)
    return ani


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def total_spillover_index(matrix: np.ndarray) -> float:
    """
    Compute total spillover index: sum of absolute values of off-diagonal elements.

    Args:
        matrix: (N, N) array

    Returns:
        Total spillover index (scalar)
    """
    N = matrix.shape[0]
    mask = ~np.eye(N, dtype=bool)
    return np.sum(np.abs(matrix[mask]))


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def _load_payload_npz(path: str) -> Dict[str, Any]:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Could not find params file: {p}")
    if p.suffix.lower() != ".npz":
        raise ValueError("path must be a '.npz' file (only supported format).")
    data = np.load(p, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _cumulative_spectral_power(
    abs_eigs_sorted: np.ndarray,
) -> Tuple[float, float, float]:
    if abs_eigs_sorted.size == 0:
        return np.nan, np.nan, np.nan
    abs_eigs_sq = np.square(abs_eigs_sorted)
    denom = abs_eigs_sq.sum()
    if denom <= 0.0:
        return np.nan, np.nan, np.nan

    def _csp(k: int) -> float:
        if abs_eigs_sq.size < k:
            return np.nan
        return abs_eigs_sq[:k].sum() / denom

    return _csp(1), _csp(2), _csp(3)


def _von_neumann_entropy(abs_eigs: np.ndarray) -> float:
    denom = abs_eigs.sum()
    if denom <= 0.0:
        return np.nan
    p = abs_eigs / denom
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))


def _principal_ipr(eigvecs: np.ndarray, idx: Optional[int]) -> float:
    if idx is None:
        return np.nan
    v = np.asarray(eigvecs[:, idx])
    v_abs = np.abs(v)
    norm = np.linalg.norm(v_abs)
    if norm <= 0.0:
        return np.nan
    v_norm = v_abs / norm
    return float(np.sum(v_norm**4))


def _katz_max(W: np.ndarray, alpha: float = 0.9) -> float:
    N = W.shape[0]
    try:
        katz_vec = np.linalg.solve(np.eye(N) - alpha * W.T, np.ones(N))
    except np.linalg.LinAlgError:
        return np.nan
    return float(np.max(katz_vec))


def _net_cancellation(
    W_star: np.ndarray, W_layers: List[np.ndarray], delta: np.ndarray
) -> float:
    numerator = np.sum(np.abs(W_star))
    denom = 0.0
    for k, W in enumerate(W_layers):
        denom += delta[k] * np.sum(np.abs(W))
    if denom <= 0.0:
        return np.nan
    return float(1.0 - numerator / denom)


def systemic_metrics(path: str) -> None:
    """
    Plot systemic metrics for W* from a saved Multiplex fit.

    Args:
        path: path to .npz results file
    """
    payload = _load_payload_npz(path)

    required = ("dates", "W_star", "delta_hat")
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"Payload missing keys: {', '.join(missing)}")

    dates = pd.to_datetime(payload["dates"])
    W_star_store = np.asarray(payload["W_star"], dtype=float)
    delta_hat = np.asarray(payload["delta_hat"], dtype=float)

    if delta_hat.ndim == 1:
        delta_hat = delta_hat[:, None]
    if W_star_store.ndim != 3:
        raise ValueError(
            f"Expected W_star with shape (T,N,N). Got {W_star_store.shape=}"
        )

    T, N, N2 = W_star_store.shape
    if N != N2:
        raise ValueError(f"Expected square W_star matrices; got {W_star_store.shape=}")
    if len(dates) != T:
        raise ValueError(f"dates length must match T. Got {len(dates)=} vs {T=}")
    if delta_hat.shape[0] != T:
        raise ValueError(
            f"delta_hat length must match T. Got {delta_hat.shape[0]=} vs {T=}"
        )

    num_layers = delta_hat.shape[1]
    layer_keys = [f"W{k + 1}" for k in range(num_layers)]
    missing_layers = [k for k in layer_keys if k not in payload]
    if missing_layers:
        raise KeyError(f"Payload missing layer keys: {', '.join(missing_layers)}")

    W_layers = [np.asarray(payload[k], dtype=float) for k in layer_keys]
    for idx, W in enumerate(W_layers, start=1):
        if W.shape != (T, N, N):
            raise ValueError(f"W{idx} must have shape {(T, N, N)}. Got {W.shape=}.")

    net_cancel = np.full(T, np.nan)
    csp1 = np.full(T, np.nan)
    csp2 = np.full(T, np.nan)
    csp3 = np.full(T, np.nan)
    entropy = np.full(T, np.nan)
    non_normal = np.full(T, np.nan)
    ipr = np.full(T, np.nan)
    katz = np.full(T, np.nan)

    for t in range(T):
        W = W_star_store[t]
        eigvals, eigvecs = np.linalg.eig(W)
        abs_eigs = np.abs(eigvals)
        order = np.argsort(abs_eigs)[::-1]
        abs_sorted = abs_eigs[order]

        csp1[t], csp2[t], csp3[t] = _cumulative_spectral_power(abs_sorted)
        entropy[t] = _von_neumann_entropy(abs_eigs)

        lambda_max = abs_sorted[0] if abs_sorted.size > 0 else np.nan
        sigma_max = np.linalg.norm(W, ord=2)
        if lambda_max > 0.0 and np.isfinite(sigma_max):
            non_normal[t] = float(sigma_max / lambda_max)

        idx = int(order[0]) if order.size > 0 else None
        ipr[t] = _principal_ipr(eigvecs, idx)

        W_layers_t = [layer[t] for layer in W_layers]
        net_cancel[t] = _net_cancellation(W, W_layers_t, delta_hat[t])
        katz[t] = _katz_max(W)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    axes[0].plot(dates, net_cancel, linewidth=1.5, color="#1F77B4")
    axes[0].set_title("Net Cancellation (W*)", fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Net Cancellation")
    axes[0].grid(True, alpha=0.3)

    csp_colors = {"low": "#FFDF2C", "mid": "#FF7F0E", "high": "#D62728"}

    axes[1].fill_between(
        dates, 0, csp1, color=csp_colors["low"], alpha=0.7, linewidth=0
    )
    axes[1].fill_between(
        dates, csp1, csp2, color=csp_colors["mid"], alpha=0.7, linewidth=0
    )
    axes[1].fill_between(
        dates, csp2, csp3, color=csp_colors["high"], alpha=0.7, linewidth=0
    )

    axes[1].plot(dates, csp1, linewidth=1, label="CSP 1", color=csp_colors["low"])
    axes[1].plot(dates, csp2, linewidth=1, label="CSP 1-2", color=csp_colors["mid"])
    axes[1].plot(dates, csp3, linewidth=1, label="CSP 1-3", color=csp_colors["high"])
    axes[1].set_title("Cumulative Spectral Power (|λ|^2)", fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("CSP")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(dates, entropy, linewidth=1.5, color="#9467BD")
    axes[2].set_title("Spectral Von Neumann Entropy", fontweight="bold")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Entropy")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(dates, non_normal, linewidth=1.5, color="#D62728")
    axes[3].set_title("Non-normality Index", fontweight="bold")
    axes[3].set_xlabel("Date")
    axes[3].set_ylabel("Nu")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(dates, ipr, linewidth=1.5, color="#8C564B")
    axes[4].set_title("IPR (principal eigenvector)", fontweight="bold")
    axes[4].set_xlabel("Date")
    axes[4].set_ylabel("IPR")
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(dates, katz, linewidth=1.5, color="#2CA02C")
    axes[5].set_title("Katz Centrality (max, α=0.90)", fontweight="bold")
    axes[5].set_xlabel("Date")
    axes[5].set_ylabel("Centrality")
    axes[5].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_deltas(
    multiplex_params: str,
    *,
    delta_labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot delta_hat weights over time from a saved Multiplex fit.

    Args:
        multiplex_params: path to .npz file
        delta_labels: optional list of labels for each delta series

    Returns:
        fig, ax
    """
    from pathlib import Path

    data = np.load(Path(multiplex_params), allow_pickle=False)
    payload = {k: data[k] for k in data.files}

    if "dates" not in payload or "delta_hat" not in payload:
        raise KeyError("Payload must contain keys: 'dates' and 'delta_hat'.")

    dates = pd.to_datetime(payload["dates"])
    deltas = np.asarray(payload["delta_hat"], dtype=float)

    if deltas.ndim == 1:
        deltas = deltas[:, None]
    if len(dates) != deltas.shape[0]:
        raise ValueError(
            "dates length must match delta_hat first dimension. "
            f"Got {len(dates)=} vs {deltas.shape[0]=}"
        )

    num_layers = deltas.shape[1]
    if delta_labels is None:
        delta_labels = [f"δ_{i + 1}" for i in range(num_layers)]
    if len(delta_labels) != num_layers:
        raise ValueError(
            f"delta_labels length must be {num_layers}. Got {len(delta_labels)}."
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(num_layers):
        ax.plot(dates, deltas[:, i], linewidth=1.5, label=delta_labels[i])

    ax.set_title("Delta Weights Over Time", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Delta")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_spillover_indices(
    multiplex_params: str,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot spillover indices over time for matrices A, B, kron(B,A), and W*.
    Includes positive and negative parts of W*.

    Args:
        multiplex_params: path to .npz file

    Returns:
        fig, axes
    """
    from pathlib import Path

    p = Path(multiplex_params)
    data = np.load(p, allow_pickle=False)
    payload = {k: data[k] for k in data.files}

    dates = pd.to_datetime(payload["dates"])
    A_store = payload["A"]
    B_store = payload["B"]
    W_star_store = payload["W_star"]

    T = len(dates)
    spillover_A = np.zeros(T)
    spillover_B = np.zeros(T)
    spillover_kron = np.zeros(T)
    spillover_W = np.zeros(T)
    spillover_W_pos = np.zeros(T)
    spillover_W_neg = np.zeros(T)

    for t in range(T):
        A = A_store[t]
        B = B_store[t]
        W = W_star_store[t]

        spillover_A[t] = total_spillover_index(A)
        spillover_B[t] = total_spillover_index(B)
        spillover_kron[t] = total_spillover_index(np.kron(B, A))
        spillover_W[t] = total_spillover_index(W)
        W_pos = np.where(W > 0.0, W, 0.0)
        W_neg = np.where(W < 0.0, W, 0.0)
        spillover_W_pos[t] = total_spillover_index(W_pos)
        spillover_W_neg[t] = total_spillover_index(W_neg)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes = axes.flatten()

    axes[0].plot(dates, spillover_A, linewidth=1.5, color="#1F77B4")
    axes[0].set_title("Spillover Index: A", fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Total Spillover")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dates, spillover_B, linewidth=1.5, color="#FF7F0E")
    axes[1].set_title("Spillover Index: B", fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Total Spillover")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dates, spillover_kron, linewidth=1.5, color="#2CA02C")
    axes[2].set_title("Spillover Index: kron(B,A)", fontweight="bold")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Total Spillover")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(dates, spillover_W, linewidth=1.5, color="#D62728")
    axes[3].set_title("Spillover Index: W*", fontweight="bold")
    axes[3].set_xlabel("Date")
    axes[3].set_ylabel("Total Spillover")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(dates, spillover_W_pos, linewidth=1.5, color="#9467BD")
    axes[4].set_title("Spillover Index: W* (positive)", fontweight="bold")
    axes[4].set_xlabel("Date")
    axes[4].set_ylabel("Total Spillover")
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(dates, spillover_W_neg, linewidth=1.5, color="#8C564B")
    axes[5].set_title("Spillover Index: W* (negative)", fontweight="bold")
    axes[5].set_xlabel("Date")
    axes[5].set_ylabel("Total Spillover")
    axes[5].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    return fig, axes


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def multiplex_4_layer_plots(
    multiplex_params: str,
    *,
    alpha: float = 0.9,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot spillover indices for the four-layer multiplex and W*,
    plus Katz centrality over time.

    Args:
        multiplex_params: path to .npz file produced by the 4-layer trainer
        alpha: Katz centrality damping factor

    Returns:
        fig, axes
    """
    from pathlib import Path

    data = np.load(Path(multiplex_params), allow_pickle=False)
    payload = {k: data[k] for k in data.files}

    required_keys = ("dates", "W1", "W2", "W3", "W4", "W_star")
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise KeyError(f"Payload missing keys: {', '.join(missing)}")

    dates = pd.to_datetime(payload["dates"])
    W_layers = [
        np.asarray(payload["W1"], dtype=float),
        np.asarray(payload["W2"], dtype=float),
        np.asarray(payload["W3"], dtype=float),
        np.asarray(payload["W4"], dtype=float),
    ]
    W_star_store = np.asarray(payload["W_star"], dtype=float)

    if W_star_store.ndim != 3:
        raise ValueError(
            f"Expected W_star with shape (T,N,N). Got {W_star_store.shape=}"
        )
    T, N, N2 = W_star_store.shape
    if N != N2:
        raise ValueError(f"Expected square W_star matrices; got {W_star_store.shape=}")
    if len(dates) != T:
        raise ValueError(f"dates length must match T. Got {len(dates)=} vs {T=}")

    for idx, W in enumerate(W_layers, start=1):
        if W.ndim != 3:
            raise ValueError(f"Expected W{idx} with shape (T,N,N). Got {W.shape=}")
        if W.shape != (T, N, N):
            raise ValueError(
                f"W{idx} must match W_star shape {(T, N, N)}. Got {W.shape=}."
            )

    layer_spillovers = np.zeros((4, T))
    w_spillover = np.zeros(T)
    katz = np.zeros(T)

    for t in range(T):
        for i, W in enumerate(W_layers):
            layer_spillovers[i, t] = total_spillover_index(W[t])
        W = W_star_store[t]
        w_spillover[t] = total_spillover_index(W)
        katz_vec = np.linalg.solve(np.eye(N) - alpha * W.T, np.ones(N))
        katz[t] = katz_vec.max()

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    axes = axes.flatten()

    colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD"]
    for i in range(4):
        axes[i].plot(dates, layer_spillovers[i], linewidth=1.5, color=colors[i])
        axes[i].set_title(
            f"Spillover Index: Layer {i + 1} (W{i + 1})", fontweight="bold"
        )
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel("Total Spillover")
        axes[i].grid(True, alpha=0.3)

    axes[4].plot(dates, w_spillover, linewidth=1.5, color="#D62728")
    axes[4].set_title("Spillover Index: W*", fontweight="bold")
    axes[4].set_xlabel("Date")
    axes[4].set_ylabel("Total Spillover")
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(dates, katz, linewidth=1.5, color="#8C564B")
    axes[5].set_title(f"Katz Centrality (max, α={alpha:.2f})", fontweight="bold")
    axes[5].set_xlabel("Date")
    axes[5].set_ylabel("Centrality")
    axes[5].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    return fig, axes


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def network_diagnostic(
    multiplex_params: str,
    *,
    tau_w: float = 0.10,
    alpha: float = 0.9,
    ewma_periods: int = 12,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot network diagnostics from W*:
    - net in-strength (max row-sum)
    - net out-strength (max column-sum)
    - density of strong links
    - Katz centrality (max)

    Args:
        multiplex_params: path to .npz file
        tau_w: threshold for strong links
        alpha: Katz centrality damping factor
        ewma_periods: number of periods to use for EWMA
    """
    from pathlib import Path

    data = np.load(Path(multiplex_params), allow_pickle=False)
    payload = {k: data[k] for k in data.files}

    dates = pd.to_datetime(payload["dates"])
    W_star_store = payload["W_star"]
    T, N = len(dates), W_star_store.shape[1]

    # Initialize metrics
    metrics = {
        "in_pos": np.zeros(T),
        "in_neg": np.zeros(T),
        "out_pos": np.zeros(T),
        "out_neg": np.zeros(T),
        "density": np.zeros(T),
        "katz": np.zeros(T),
    }

    # Compute metrics
    for t in range(T):
        W = W_star_store[t]
        W_off = W - np.diag(np.diag(W))

        W_pos = np.maximum(W_off, 0)
        W_neg = np.maximum(-W_off, 0)

        metrics["in_pos"][t] = W_pos.sum(axis=1).max()
        metrics["in_neg"][t] = W_neg.sum(axis=1).max()
        metrics["out_pos"][t] = W_pos.sum(axis=0).max()
        metrics["out_neg"][t] = W_neg.sum(axis=0).max()
        metrics["density"][t] = (np.abs(W_off) > tau_w).sum() / (N * (N - 1))

        katz_vec = np.linalg.solve(np.eye(N) - alpha * W.T, np.ones(N))
        metrics["katz"][t] = katz_vec.max()

    # Helper function for EWMA
    def ewma_stats(series):
        ewm = pd.Series(series).ewm(span=ewma_periods, adjust=False)
        return ewm.mean().to_numpy(), ewm.std(bias=False).to_numpy()

    # Helper function to plot strength with EWMA bands
    def plot_strength(ax, metric_pos, metric_neg, title, colors=None):
        if colors is None:
            colors = ["C0", "red"]

        # Raw data
        ax.plot(
            dates,
            metric_pos,
            linewidth=0.85,
            alpha=0.7,
            label="Positive",
            color=colors[0],
        )
        ax.plot(
            dates,
            metric_neg,
            linewidth=0.85,
            alpha=0.7,
            label="Negative",
            color=colors[1],
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Strength")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        return colors

    def plot_ewma_bands(ax, metric_pos, metric_neg, title, colors):
        mean_pos, std_pos = ewma_stats(metric_pos)
        mean_neg, std_neg = ewma_stats(metric_neg)

        # EWMA lines
        ax.plot(
            dates,
            mean_pos,
            linewidth=1,
            alpha=0.9,
            color=colors[0],
            label="Positive EWMA",
        )
        ax.plot(
            dates,
            mean_neg,
            linewidth=1,
            alpha=0.9,
            color=colors[1],
            label="Negative EWMA",
        )

        # Shaded ±1σ bands
        ax.fill_between(
            dates, mean_pos - std_pos, mean_pos + std_pos, color=colors[0], alpha=0.12
        )
        ax.fill_between(
            dates, mean_neg - std_neg, mean_neg + std_neg, color=colors[1], alpha=0.12
        )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Strength")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Plot strength metrics
    colors_in = plot_strength(
        axes[0], metrics["in_pos"], metrics["in_neg"], "Net In-Strength (max row-sum)"
    )
    plot_strength(
        axes[1],
        metrics["out_pos"],
        metrics["out_neg"],
        "Net Out-Strength (max column-sum)",
    )

    # Plot EWMA with bands
    plot_ewma_bands(
        axes[2],
        metrics["in_pos"],
        metrics["in_neg"],
        "EWMA Net In-Strength (±1σ)",
        colors_in,
    )
    plot_ewma_bands(
        axes[3],
        metrics["out_pos"],
        metrics["out_neg"],
        "EWMA Net Out-Strength (±1σ)",
        ["C0", "red"],
    )

    # Density and Katz
    axes[4].plot(dates, metrics["density"], linewidth=1, color="#2CA02C")
    axes[4].set_title(f"Density of Strong Links (τ={tau_w:.2f})", fontweight="bold")
    axes[4].set_xlabel("Date")
    axes[4].set_ylabel("Density")
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(dates, metrics["katz"], linewidth=1, color="#D62728")
    axes[5].set_title(f"Katz Centrality (max, α={alpha:.2f})", fontweight="bold")
    axes[5].set_xlabel("Date")
    axes[5].set_ylabel("Centrality")
    axes[5].grid(True, alpha=0.3)

    # Format x-axis for all subplots
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    return fig, axes
