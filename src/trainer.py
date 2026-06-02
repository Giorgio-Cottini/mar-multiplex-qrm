import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional, Union
from .models import MAR, Multiplex
from .loader import to_MAR_matrix, to_Multiplex_matrices, ASSET_GRID


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #
def fit_MAR_Multiplex(
    returns: pd.DataFrame,
    volatilities: pd.DataFrame,
    factors: pd.DataFrame,
    context_length: int = 252,
    step: int = 5,
    save_path: str = "multiplex_params_mar.npz",
) -> dict:
    """
    Fit the MAR-Multiplex model, given returns, volatilities and factors.
    It fits the MAR model on returns and volatilities, then compute W = kron(A, B) and fit the Multiplex network on the W obtained from returns and factors.
    Args:
        returns: pd.DataFrame - Returns
        volatilities: pd.DataFrame - Volatilities
        factors: pd.DataFrame - Factors
        context_length: int - Context length
        step: int - Step size
        save_path: str - Save path
    Returns:
        dict - Results with dates, A, B, W1, W2, W_star, delta_hat, rho_hat, beta_hat, sigma_sq
    """
    results = []
    returns_model = MAR()
    volatilities_model = MAR()
    total_assets = sum(len(row) for row in ASSET_GRID)
    network = Multiplex(num_layers=2, num_assets=total_assets)
    # Storage for later reconstruction (always saved)
    save_path = Path(save_path)
    if save_path.suffix.lower() != ".npz":
        raise ValueError("save_path must end with '.npz' (only supported format).")
    n_steps = len(range(0, len(returns) - context_length + 1, step))
    N = total_assets
    K = factors.shape[1]
    stores = {
        "dates": np.empty(n_steps, dtype="datetime64[ns]"),
        "A": np.empty((n_steps, N, N), dtype=float),
        "B": np.empty((n_steps, N, N), dtype=float),
        "W1": np.empty((n_steps, N, N), dtype=float),
        "W2": np.empty((n_steps, N, N), dtype=float),
        "W_star": np.empty((n_steps, N, N), dtype=float),
        "delta_hat": np.empty((n_steps, network.num_layers), dtype=float),
        "rho_hat": np.empty((n_steps, N), dtype=float),
        "beta_hat": np.empty((n_steps, K, N), dtype=float),
        "sigma_sq": np.empty(n_steps, dtype=float),
    }
    for i in tqdm(range(0, len(returns) - context_length + 1, step)):
        R = returns.iloc[i : i + context_length]
        V = volatilities.iloc[i : i + context_length]
        F = factors.iloc[i : i + context_length]
        # De-mean returns
        R_noMean = R - R.mean()
        # Compute log volatilities and de-mean
        logV_noMean = np.log(V) - np.log(V).mean()
        # Convert to matrix
        X_batch = to_MAR_matrix(R_noMean, mapping=ASSET_GRID)
        V_batch = to_MAR_matrix(logV_noMean, mapping=ASSET_GRID)
        R_batch, F_batch = to_Multiplex_matrices(R_noMean, F)
        # Initialize or set the matrices
        if i == 0:
            returns_model.generate_starting_matrices(X_batch)
            volatilities_model.generate_starting_matrices(V_batch)
        # Fit the MAR model on returns and volatilities
        returns_model.fit(X_batch, max_iter=500, patience=2)
        volatilities_model.fit(V_batch, max_iter=500, patience=2)
        # Set adjacency matrices in the Multiplex network
        W_V = volatilities_model.get_matrices()["W"]
        W_R = returns_model.get_matrices()["W"]
        network.set_adjacency_matrices([W_V, W_R])
        network_results = network.fit(R_batch, F_batch)
        # Store Results
        k = len(results)
        stores["dates"][k] = R.index[-1]
        stores["A"][k] = W_V  # yeah its a stretch of notation
        stores["B"][k] = W_R  # same here
        stores["W1"][k] = network.get_adjacency_matrices()[0]
        stores["W2"][k] = network.get_adjacency_matrices()[1]
        stores["W_star"][k] = network_results["W_star"]
        stores["delta_hat"][k] = network_results["delta_hat"]
        stores["rho_hat"][k] = network_results["rho_hat"]
        stores["beta_hat"][k] = network_results["beta_hat"]
        stores["sigma_sq"][k] = network_results["sigma_sq"]
        results.append(network_results)
    np.savez_compressed(save_path, **stores)
    return results


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #
def fit_returns_MAR(
    df: pd.DataFrame,
    model: MAR,
    context_length: int = 252,
    mapping: List[List[str]] = ASSET_GRID,
    step: int = 1,
    full_diagnostics: bool = False,
) -> pd.DataFrame:
    """
    Fits the MAR model on a rolling window of data.
    Args:
        df: DataFrame with datetime index and asset returns columns.
        model: Instance of your MAR class.
        context_length: The size of the rolling window (T).
        mapping: 2D list of asset tickers defining the matrix structure (m x n).
        step: Step size for rolling window (default: 1).
        full_diagnostics: bool, default: False
            Whether to compute full diagnostics of the model.
    Returns:
        pd.DataFrame: Results with window_end_date, final_ll, and iterations.
    """
    # Preliminary checks and setup
    results = []
    for i in tqdm(range(0, len(df) - context_length + 1, step)):
        # Get the current window
        window = df.iloc[i : i + context_length]  # Shape: (context_length, assets)
        # De-mean window
        window = window - window.mean()
        # Convert to matrix
        X_batch = to_MAR_matrix(window, mapping=mapping)  # Shape: (m, n, T)
        # Initialize or set the matrices
        if i == 0:
            model.generate_starting_matrices(X_batch)
        # Fit the model
        log_likelihoods, iters, rho_A_series, rho_B_series, rho_W_series = model.fit(
            X_batch, max_iter=500, patience=10
        )
        # Store Results
        result_dict = {
            "window_start_date": window.index[0],
            "window_end_date": window.index[-1],
            "final_ll": log_likelihoods[-1],
            "iterations": iters,
            "rho_A": rho_A_series[-1],
            "rho_B": rho_B_series[-1],
            "rho_W": rho_W_series[-1],
        }
        if full_diagnostics:
            result_dict.update(model.full_diagnostics())
        results.append(result_dict)
    return pd.DataFrame(results)


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #
def quantile_dependence_matrix(
    df: pd.DataFrame, quantile: float, threshold: float = 0.0
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Compute Quantile Dependence Matrix (QDM) following tail dependence literature.
    The QDM captures co-movement in extreme events (e.g., joint crashes at q=0.05).

    Args:
        df: DataFrame (T x N) - Asset returns or other time series
        quantile: Quantile level (e.g., 0.05 for 5% left tail, 0.95 for right tail)
        threshold: Minimum correlation value to keep (values below are set to 0)

    Returns:
        Quantile dependence matrix (N x N).
    """
    # Compute indicator matrix
    indicators = df.le(df.quantile(quantile)).astype(float)
    # Compute QDM
    qdm = np.nan_to_num(indicators.corr().to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    # Enforce thresholds
    if threshold > 0:
        qdm[np.abs(qdm) < threshold] = 0.0

    return qdm


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def get_qdm_rolling(
    returns: pd.DataFrame,
    window: int,
    quantile: float,
    threshold: float = 0.0,
    return_end_dates: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, pd.Index]]:
    """
    Compute rolling Quantile Dependence Matrices (QDMs).

    Args:
        returns: DataFrame (T x N) - Asset returns
        window: Rolling window size (e.g., 252 for 1 year)
        quantile: Quantile level (0.05 for left tail, 0.95 for right tail)
        threshold: Minimum absolute correlation to keep (sparse networks if > 0)
        return_end_dates: If True, also return the window end dates (Index of length T_windows).

    Returns:
        If return_end_dates is False:
            Rolling QDM matrices as a MAR-ready tensor X with shape (N, N, T_windows).
        If return_end_dates is True:
            (X, end_dates) where end_dates[k] is the end date of the k-th rolling window.
    """
    ends = range(window, len(returns) + 1)
    n = len(ends)
    N = returns.shape[1]
    X = np.zeros((N, N, n), dtype=float)
    end_dates: list = []
    for k, end in enumerate(tqdm(ends, desc="Computing rolling QDM")):
        X[:, :, k] = quantile_dependence_matrix(
            returns.iloc[end - window : end], quantile, threshold
        )
        # Preserve the window end index if possible (typically a Timestamp)
        end_dates.append(returns.index[end - 1])

    if return_end_dates:
        return X, pd.Index(end_dates)
    return X


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def fit_qdm_MAR(
    qdms: Union[np.ndarray, Tuple[np.ndarray, pd.Index]],
    model: MAR,
    context_length: int,
    step: int = 5,
    full_diagnostics: bool = False,
) -> pd.DataFrame:
    """
    Fit MAR on a rolling window of QDM matrices.

    Args:
        qdms: Union[np.ndarray, Tuple[np.ndarray, pd.Index]] - QDM matrices
        model: MAR - MAR model
        context_length: int - Context length
        step: int - Step size
        full_diagnostics: bool - Whether to compute full diagnostics of the model
    Returns:
        pd.DataFrame - Results with window_start_date, window_end_date, final_ll, iterations, rho_A, rho_B, rho_W

    Notes:
    - Expected qdms shape is (N, N, T), where T is the number of time windows.
    - MAR is a one-step model: X_t ≈ A X_{t-1} B, so each window provides a time series.
    """

    qdm_dates: Optional[pd.Index] = None
    if isinstance(qdms, tuple):
        if len(qdms) != 2:
            raise ValueError("qdms tuple must be (qdms_array, dates_index)")
        qdms, qdm_dates = qdms

    results: List[dict] = []
    for i in tqdm(
        range(0, qdms.shape[2] - context_length + 1, step),
        desc="Fitting MAR on rolling QDM",
    ):
        X_batch = qdms[:, :, i : i + context_length]  # (N, N, context_length)

        # Demean *within the window* (avoids look-ahead leakage vs global demeaning).
        X_batch = X_batch - X_batch.mean(axis=2, keepdims=True)

        if i == 0:
            model.generate_starting_matrices(X_batch)

        log_likelihoods, iters, rho_A_series, rho_B_series, rho_W_series = model.fit(
            X_batch, max_iter=500, patience=1
        )  # it usually converges in ~10 iterations

        if qdm_dates is None:
            window_start = i
            window_end = i + context_length
        else:
            window_start = qdm_dates[i]
            window_end = qdm_dates[i + context_length - 1]

        result_dict = {
            "window_start_date": window_start,
            "window_end_date": window_end,
            "final_ll": log_likelihoods[-1],
            "iterations": iters,
            "rho_A": rho_A_series[-1],
            "rho_B": rho_B_series[-1],
            "rho_W": rho_W_series[-1],
        }
        if full_diagnostics:
            result_dict.update(model.full_diagnostics())
        results.append(result_dict)

    return pd.DataFrame(results)


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def align_inputs(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    qdms: np.ndarray,
    qdm_end_dates: pd.Index,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Index]:
    """
    Short alignment helper: make a strict 1:1 mapping of dates across returns, factors,
    and the QDM time axis (qdms[:, :, t] <-> qdm_end_dates[t]).

    It trims everything to the common dates (which also makes them start at the latest
    starting source).

    Args:
        returns: pd.DataFrame - Returns
        factors: pd.DataFrame - Factors
        qdms: np.ndarray - QDM matrices
        qdm_end_dates: pd.Index - QDM end dates
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Index] - Returns, factors, qdms, common
    """
    qdm_end_dates = pd.Index(qdm_end_dates)
    common = returns.index.intersection(factors.index).intersection(qdm_end_dates)
    if len(common) == 0:
        raise ValueError(
            "No overlapping dates between returns, factors and qdm_end_dates."
        )
    pos = qdm_end_dates.get_indexer(common)
    returns = returns.loc[common]
    factors = factors.loc[common]
    qdms = qdms[:, :, pos]
    return returns, factors, qdms, common


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def fit_qdm_Multiplex_MAR_split(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    qdms: Tuple[np.ndarray, pd.Index],
    context_length: int,
    step: int = 5,
    save_path: str = "multiplex_params_split.npz",
) -> None:
    # Require explicit QDM dates (no "date-less" mode).
    qdms_array, qdm_end_dates = qdms
    returns, factors, qdms_array, qdm_end_dates = align_inputs(
        returns, factors, qdms_array, qdm_end_dates
    )
    # Initialize the Multiplex network
    network = Multiplex(num_layers=4, num_assets=qdms_array.shape[0])

    # Storage for later reconstruction (always saved)
    save_path = Path(save_path)
    if save_path.suffix.lower() != ".npz":
        raise ValueError("save_path must end with '.npz' (only supported format).")
    # Compute the number of steps
    n_steps = len(range(0, qdms_array.shape[2] - context_length + 1, step))
    N = qdms_array.shape[0]
    K = factors.shape[1]

    # Pre-allocate stores (keeps loop tidy, still fast for 1k-5k windows).
    stores = {
        "dates": np.empty(n_steps, dtype="datetime64[ns]"),
        "A": np.empty((n_steps, N, N), dtype=float),
        "B": np.empty((n_steps, N, N), dtype=float),
        "W1": np.empty((n_steps, N, N), dtype=float),
        "W2": np.empty((n_steps, N, N), dtype=float),
        "W3": np.empty((n_steps, N, N), dtype=float),
        "W4": np.empty((n_steps, N, N), dtype=float),
        "W_star": np.empty((n_steps, N, N), dtype=float),
        "delta_hat": np.empty((n_steps, network.num_layers), dtype=float),
        "rho_hat": np.empty((n_steps, N), dtype=float),
        "beta_hat": np.empty((n_steps, K, N), dtype=float),
        "sigma_sq": np.empty(n_steps, dtype=float),
    }
    # Reuse a single MAR instance across windows.
    mar_model = MAR()

    # Fit the model
    for k, i in enumerate(
        tqdm(range(0, qdms_array.shape[2] - context_length + 1, step))
    ):
        # Get the current window
        X_batch = qdms_array[:, :, i : i + context_length]
        dates_window = qdm_end_dates[i : i + context_length]
        R_batch = returns.loc[dates_window]
        F_batch = factors.loc[dates_window]
        # Demean within window (consistent with fit_qdm_MAR)
        X_batch = X_batch - X_batch.mean(axis=2, keepdims=True)
        # Fit MAR on the current window and extract matrices
        # Initialize the matrices if the first window
        if i == 0:
            mar_model.generate_starting_matrices(X_batch)
        # Fit the model
        mar_model.fit(X_batch, max_iter=500, patience=1)
        # Set the adjacency matrices
        model_matrices = mar_model.get_matrices()
        # Extract the positive and negative parts of the matrix A
        A_pos = np.where(model_matrices["A"] > 0, model_matrices["A"], 0)
        A_neg = np.where(model_matrices["A"] < 0, model_matrices["A"], 0)
        # Extract the positive and negative parts of the matrix B
        B_pos = np.where(model_matrices["B"] > 0, model_matrices["B"], 0)
        B_neg = np.where(model_matrices["B"] < 0, model_matrices["B"], 0)
        network.set_adjacency_matrices([A_pos, A_neg, B_pos, B_neg], row_normalize=True)
        # Fit the network
        R_np = (R_batch - R_batch.mean(axis=0)).to_numpy().T
        F_np = F_batch.to_numpy()
        network_results = network.fit(R_np, F_np, max_iter=500, ftol=1e-7)

        # Store per-window results
        stores["dates"][k] = dates_window[-1]
        stores["A"][k] = model_matrices["A"]
        stores["B"][k] = model_matrices["B"]
        stores["W1"][k] = network.get_adjacency_matrices()[0]
        stores["W2"][k] = network.get_adjacency_matrices()[1]
        stores["W3"][k] = network.get_adjacency_matrices()[2]
        stores["W4"][k] = network.get_adjacency_matrices()[3]
        stores["W_star"][k] = network_results["W_star"]
        stores["delta_hat"][k] = network_results["delta_hat"]
        stores["rho_hat"][k] = network_results["rho_hat"]
        stores["beta_hat"][k] = network_results["beta_hat"]
        stores["sigma_sq"][k] = network_results["sigma_sq"]

    np.savez_compressed(save_path, **stores)


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def fit_qdm_Multiplex_MAR(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    qdms: Tuple[np.ndarray, pd.Index],
    context_length: int,
    step: int = 5,
    save_path: str = "multiplex_params.npz",
) -> None:
    """
    Fit the QDM-Multiplex-MAR model.
    Args:
        returns: pd.DataFrame - Returns
        factors: pd.DataFrame - Factors
        qdms: Tuple[np.ndarray, pd.Index] - QDM matrices and end dates
        context_length: int - Context length
        step: int - Step size
        save_path: str - Save path
    """
    # Require explicit QDM dates (no "date-less" mode).
    qdms_array, qdm_end_dates = qdms
    returns, factors, qdms_array, qdm_end_dates = align_inputs(
        returns, factors, qdms_array, qdm_end_dates
    )
    # Initialize the Multiplex network
    network = Multiplex(num_layers=2, num_assets=qdms_array.shape[0])

    # Storage for later reconstruction (always saved)
    save_path = Path(save_path)
    if save_path.suffix.lower() != ".npz":
        raise ValueError("save_path must end with '.npz' (only supported format).")
    # Compute the number of steps
    n_steps = len(range(0, qdms_array.shape[2] - context_length + 1, step))
    N = qdms_array.shape[0]
    K = factors.shape[1]

    # Pre-allocate stores (keeps loop tidy, still fast for 1k-5k windows).
    stores = {
        "dates": np.empty(n_steps, dtype="datetime64[ns]"),
        "A": np.empty((n_steps, N, N), dtype=float),
        "B": np.empty((n_steps, N, N), dtype=float),
        "W1": np.empty((n_steps, N, N), dtype=float),
        "W2": np.empty((n_steps, N, N), dtype=float),
        "W_star": np.empty((n_steps, N, N), dtype=float),
        "delta_hat": np.empty((n_steps, network.num_layers), dtype=float),
        "rho_hat": np.empty((n_steps, N), dtype=float),
        "beta_hat": np.empty((n_steps, K, N), dtype=float),
        "sigma_sq": np.empty(n_steps, dtype=float),
    }

    # Reuse a single MAR instance across windows.
    mar_model = MAR()

    # Fit the model
    for k, i in enumerate(
        tqdm(range(0, qdms_array.shape[2] - context_length + 1, step))
    ):
        # Get the current window
        X_batch = qdms_array[:, :, i : i + context_length]
        dates_window = qdm_end_dates[i : i + context_length]
        R_batch = returns.loc[dates_window]
        F_batch = factors.loc[dates_window]
        # Demean within window (consistent with fit_qdm_MAR)
        X_batch = X_batch - X_batch.mean(axis=2, keepdims=True)
        # Fit MAR on the current window and extract matrices
        # Initialize the matrices if the first window
        if i == 0:
            mar_model.generate_starting_matrices(X_batch)
        # Fit the model
        mar_model.fit(X_batch, max_iter=500, patience=5)
        # Set the adjacency matrices
        model_matrices = mar_model.get_matrices()
        network.set_adjacency_matrices([model_matrices["A"], model_matrices["B"].T])
        # Fit the network
        R_np = (R_batch - R_batch.mean(axis=0)).to_numpy().T
        F_np = F_batch.to_numpy()
        network_results = network.fit(R_np, F_np, max_iter=500, ftol=1e-6)

        # Store per-window results
        stores["dates"][k] = dates_window[-1]
        stores["A"][k] = model_matrices["A"]
        stores["B"][k] = model_matrices["B"]
        stores["W1"][k] = network.get_adjacency_matrices()[0]
        stores["W2"][k] = network.get_adjacency_matrices()[1]
        stores["W_star"][k] = network_results["W_star"]
        stores["delta_hat"][k] = network_results["delta_hat"]
        stores["rho_hat"][k] = network_results["rho_hat"]
        stores["beta_hat"][k] = network_results["beta_hat"]
        stores["sigma_sq"][k] = network_results["sigma_sq"]

    np.savez_compressed(save_path, **stores)


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def fit_qdm_MultipMAR_ret_vol(
    returns: pd.DataFrame,
    volatilities: pd.DataFrame,
    factors: pd.DataFrame,
    context_length: int = 252,
    qdm_quantile: float = 0.2,
    step: int = 5,
    save_path: str = "multiplex_params_ret_vol.npz",
) -> None:
    """
    Fit the QDM-Multiplex-MAR model with returns and volatilities.
    Args:
        returns: pd.DataFrame - Returns
        volatilities: pd.DataFrame - Volatilities
        factors: pd.DataFrame - Factors
        context_length: int - Context length
        step: int - Step size
        save_path: str - Save path
    """

    qdms_array_returns, qdm_end_dates = get_qdm_rolling(
        returns, context_length, quantile=qdm_quantile, return_end_dates=True
    )
    qdms_array_volatilities, _ = get_qdm_rolling(
        volatilities, context_length, quantile=qdm_quantile, return_end_dates=True
    )

    returns, factors, qdms_array_returns, qdm_end_dates = align_inputs(
        returns, factors, qdms_array_returns, qdm_end_dates
    )
    volatilities, factors, qdms_array_volatilities, qdm_end_dates = align_inputs(
        volatilities, factors, qdms_array_volatilities, qdm_end_dates
    )
    # Initialize the Multiplex nework
    network = Multiplex(num_layers=4, num_assets=qdms_array_returns.shape[0])

    # Storage for later reconstruction (always saved)
    save_path = Path(save_path)
    if save_path.suffix.lower() != ".npz":
        raise ValueError("save_path must end with '.npz' (only supported format).")
    # Compute the number of steps
    n_steps = len(range(0, qdms_array_returns.shape[2] - context_length + 1, step))
    N = qdms_array_returns.shape[0]
    K = factors.shape[1]

    # Pre-allocate stores (keeps loop tidy, still fast for 1k-5k windows).
    stores = {
        "dates": np.empty(n_steps, dtype="datetime64[ns]"),
        "A_r": np.empty((n_steps, N, N), dtype=float),
        "B_r": np.empty((n_steps, N, N), dtype=float),
        "A_v": np.empty((n_steps, N, N), dtype=float),
        "B_v": np.empty((n_steps, N, N), dtype=float),
        "W1": np.empty((n_steps, N, N), dtype=float),
        "W2": np.empty((n_steps, N, N), dtype=float),
        "W3": np.empty((n_steps, N, N), dtype=float),
        "W4": np.empty((n_steps, N, N), dtype=float),
        "W_star": np.empty((n_steps, N, N), dtype=float),
        "delta_hat": np.empty((n_steps, network.num_layers), dtype=float),
    }

    # Reuse a single MAR instance across windows.
    returns_model = MAR()
    volatilities_model = MAR()

    # Fit the model
    for k, i in enumerate(
        tqdm(range(0, qdms_array_returns.shape[2] - context_length + 1, step))
    ):
        # Get the current window
        X_batch = qdms_array_returns[:, :, i : i + context_length]
        V_batch = qdms_array_volatilities[:, :, i : i + context_length]
        dates_window = qdm_end_dates[i : i + context_length]
        R_batch = returns.loc[dates_window]
        F_batch = factors.loc[dates_window]
        # Demean within window (consistent with fit_qdm_MAR)
        X_batch = X_batch - X_batch.mean(axis=2, keepdims=True)
        V_batch = V_batch - V_batch.mean(axis=2, keepdims=True)
        # Fit MAR on the current window and extract matrices
        # Initialize the matrices if the first window
        if i == 0:
            returns_model.generate_starting_matrices(X_batch)
            volatilities_model.generate_starting_matrices(V_batch)
        # Fit the model
        returns_model.fit(X_batch, max_iter=500, patience=1)
        volatilities_model.fit(V_batch, max_iter=500, patience=1)
        # Set the adjacency matrices
        returns_matrices = returns_model.get_matrices()
        volatilities_matrices = volatilities_model.get_matrices()
        network.set_adjacency_matrices(
            [
                returns_matrices["A"],
                returns_matrices["B"].T,
                volatilities_matrices["A"],
                volatilities_matrices["B"].T,
            ]
        )
        # Fit the network
        R_np = (R_batch - R_batch.mean(axis=0)).to_numpy().T
        F_np = F_batch.to_numpy()
        network_results = network.fit(R_np, F_np, max_iter=500, ftol=1e-6)

        # Store per-window results
        stores["dates"][k] = dates_window[-1]
        stores["A_r"][k] = returns_matrices["A"]
        stores["B_r"][k] = returns_matrices["B"]
        stores["A_v"][k] = volatilities_matrices["A"]
        stores["B_v"][k] = volatilities_matrices["B"]
        stores["W1"][k] = network.get_adjacency_matrices()[0]
        stores["W2"][k] = network.get_adjacency_matrices()[1]
        stores["W3"][k] = network.get_adjacency_matrices()[2]
        stores["W4"][k] = network.get_adjacency_matrices()[3]
        stores["W_star"][k] = network_results["W_star"]
        stores["delta_hat"][k] = network_results["delta_hat"]

    np.savez_compressed(save_path, **stores)


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #
def check_stationarity_qdm(
    qdms: np.ndarray, mode: str = "edge", i: int = 0, j: int = 1
) -> None:
    """
    ADF stationarity check for QDM tensor.

    qdms: ndarray with shape (N, N, T_windows)
    mode:
      - "edge": test the single series qdms[i, j, :]
      - "row_mean": test mean_j qdms[i, j, :]  (a per-asset summary series)
    """
    import numpy as np
    from statsmodels.tsa.stattools import adfuller
    import matplotlib.pyplot as plt

    if qdms.ndim != 3:
        raise ValueError(f"Expected qdms with ndim=3 (N,N,T), got shape {qdms.shape}")

    N, N2, _ = qdms.shape
    if N != N2:
        raise ValueError(f"Expected square QDM matrices (N,N,T); got {qdms.shape}")

    if mode == "edge":
        x = qdms[i, j, :]
        label = f"edge ({i},{j})"
    elif mode == "row_mean":
        x = qdms[i, :, :].mean(axis=0)
        label = f"row_mean (i={i})"
    else:
        raise ValueError("mode must be 'edge' or 'row_mean'")

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]  # drop NaN/inf if any

    result = adfuller(x, maxlag=12, regression="c")
    print(f"ADF test for {label}: stat={result[0]}, p-value={result[1]}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].plot(x)
    axes[0].set_title(f"Series for {label}")
    axes[1].plot(x - x.mean())
    axes[1].set_title(f"Demeaned series for {label}")
    plt.tight_layout()
    plt.show()
