import datetime
import numpy as np
import pandas as pd

# —————————————————————————————————————————————————————————————————————————————————————————————————————— #
# Each row represents a category/sector
# Columns progress from Defensive (left) to Cyclical (right)
ASSET_GRID = [
    # Row 1: Mega-cap diversified leaders
    ["KO", "JNJ", "AAPL", "IBM", "JPM", "HD", "BA"],
    # Row 2: Strong brand/product franchises
    ["PG", "MRK", "MSFT", "CSCO", "GS", "NKE", "CAT"],
    # Row 3: Platform/network businesses
    ["WMT", "UNH", "AMZN", "INTC", "AXP", "DIS", "HON"],
    # Row 4: Specialized/focused players
    ["MCD", "AMGN", "NVDA", "VZ", "SHW", "MMM", "CVX"],
]
# Column headers (economic sensitivity spectrum)
COLUMNS = [
    "Defensive Consumer",
    "Healthcare",
    "Tech Platforms",
    "Mature Tech",
    "Financials",
    "Consumer Discretionary",
    "Deep Cyclicals",
]  # B -> interaction between columns, Sigma_r -> corresponding covariance

# Row descriptions
ROWS = [
    "Mega-cap diversified leaders",
    "Strong brand/product franchises",
    "Platform/network businesses",
    "Specialized/focused players",
]  # A -> interaction between rows, Sigma_c -> corresponding covariance


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def load_formatted_data(
    filepath: str = "Dataset/dataset.xlsx",
    start_date: datetime.datetime = datetime.datetime(2004, 6, 24),
    asset_grid: list = ASSET_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load asset prices, volatilities, and FF5 factors.
    Compute log returns and align all datasets by date.

    Returns:
        log_returns: DataFrame of log returns
        volatilities: DataFrame of realized variances
        factors: DataFrame of FF5 factors
    """
    print("Loading data...")

    selected_assets = [asset for row in asset_grid for asset in row]

    # --- load prices ---
    prices = pd.read_excel(
        filepath,
        sheet_name="EoD_Prices",
        parse_dates=["Date"],
    ).set_index("Date")

    # --- load volatilities ---
    volatilities = pd.read_excel(
        filepath,
        sheet_name="Realized_Variances",
        parse_dates=["Date"],
    ).set_index("Date")

    # --- load factors ---
    factors = pd.read_excel(
        filepath,
        sheet_name="FF5",
        parse_dates=["DATE"],
    ).set_index("DATE")

    # --- filter to selected assets ---
    prices = prices[selected_assets]
    volatilities = volatilities[selected_assets]
    factors.drop(columns=["RF"], inplace=True)
    # --- compute log returns ---
    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns = np.log(prices).diff().dropna()

    # --- align datasets on dates ---
    common_index = log_returns.index.intersection(volatilities.index).intersection(
        factors.index
    )

    log_returns = log_returns.loc[common_index]
    volatilities = volatilities.loc[common_index]
    factors = factors.loc[common_index]

    # --- apply start date filter ---
    log_returns = log_returns.loc[log_returns.index >= start_date]
    volatilities = volatilities.loc[volatilities.index >= start_date]
    factors = factors.loc[factors.index >= start_date]

    return log_returns, volatilities, factors


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #
def to_MAR_matrix(X: pd.DataFrame, mapping: list[list[str]] = None) -> np.ndarray:
    """
    Convert a (T x assets) DataFrame into a matrix-valued time series (m, n, T)

    Args:
        X: DataFrame with shape (T, m*n). Columns must include the tickers in
        mapping: list[list[str]] defining the asset grid.
    Returns:
        np.ndarray with shape (m, n, T).
    """
    if mapping:
        # Flatten grid row-wise (left->right, top->bottom) to match the mapping
        order = [ticker for row in mapping for ticker in row]
        m = len(mapping)
        n = len(mapping[0])
        # Reorder the columns of the DataFrame to match the order of the mapping
        X_ordered = X.loc[:, order]
        T = len(X_ordered)
        # Shape: (T, m*n) -> (T, m, n) -> (m, n, T)
        arr = X_ordered.to_numpy(dtype=float, copy=False)
        return arr.reshape(T, m, n).transpose(1, 2, 0)
    else:
        n = int(np.sqrt(X.shape[1]))
        m = X.shape[1] // n
        return (
            X.to_numpy(dtype=float, copy=False)
            .reshape(m, n, X.shape[0])
            .transpose(1, 2, 0)
        )


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def to_Multiplex_matrices(
    R: pd.DataFrame, F: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts time-series DataFrames into the specific array formats required
    by the Multiplex Network fit function.

    Args:
        R: Returns DataFrame (Index: Date, Columns: Assets)
        F: Factors DataFrame (Index: Date, Columns: Factors)

    Returns:
        tuple containing:
        - R_arr: np.ndarray, shape (N, T) [Assets x Time]
        - F_arr: np.ndarray, shape (T, K) [Time x Factors]
    """
    # Safety check: Ensure strict time alignment before stripping indices
    if not R.index.equals(F.index):
        raise ValueError("Indices of Returns and Factors do not match.")

    return R.values.T, F.values


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


def screen_assets(
    returns: pd.DataFrame, top_n: int = 15, method: str = "variance"
) -> list:
    """
    Ultra-simple screening: pick assets with the most informative correlation patterns.

    Args:
        returns: DataFrame (T x N) - Asset returns
        top_n: Number of assets to keep
        method: 'variance' (varied correlations) or 'strength' (strong correlations)

    Returns:
        List of selected asset names
    """
    # Compute correlation matrix
    corr_matrix = returns.corr().to_numpy()
    np.fill_diagonal(corr_matrix, 0)  # Remove self-correlation

    if method == "variance":
        # Assets whose correlations vary the most across other assets
        # = more interesting/diverse relationships
        scores = np.var(corr_matrix, axis=1)

    elif method == "strength":
        # Assets with strongest average absolute correlation
        # = most connected to others
        scores = np.mean(np.abs(corr_matrix), axis=1)

    elif method == "max":
        # Assets with highest maximum correlation to any other asset
        # = strongest pairwise relationships
        scores = np.max(np.abs(corr_matrix), axis=1)

    elif method == "nonzero":
        # Assets with most significant correlations (e.g., |corr| > 0.3)
        threshold = 0.3
        scores = np.sum(np.abs(corr_matrix) > threshold, axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Rank and select
    ranking = pd.Series(scores, index=returns.columns).sort_values(ascending=False)

    return ranking.head(top_n).index.tolist()


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #
