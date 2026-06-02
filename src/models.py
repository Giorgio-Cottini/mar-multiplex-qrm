import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional


class MAR:
    """Matrix-valued Autoregressive model (MAR) class.
    The class is a direct implementation of the MAR method proposed by Chen et al. (2020).
    It contains the methods to fit the model to the data and to iteratively adjust the parameters of the model maximizing the log-likelihood.
    """

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        Sigma_c: Optional[np.ndarray] = None,
        Sigma_r: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the MAR model.
        Args:
            A: np.ndarray, shape: (m, m)
            B: np.ndarray, shape: (n, n)
            Sigma_c: np.ndarray, shape: (n, n)
            Sigma_r: np.ndarray, shape: (m, m)
        Returns:
            None
        """
        if (
            A is not None
            and B is not None
            and Sigma_c is not None
            and Sigma_r is not None
        ):
            self.set_matrices(A, B, Sigma_c, Sigma_r, True)
        else:
            self.A = None
            self.B = None
            self.W = None
            self.Sigma_c = None
            self.Sigma_r = None
            self.Sigma_W = None
            self.rho_A = None
            self.rho_B = None
            self.rho_W = None

    # —————————————————————————————————————————————————————————————————————————————————————————————————————— #
    def set_matrices(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Sigma_c: np.ndarray,
        Sigma_r: np.ndarray,
        normalize: bool = False,
    ) -> None:
        """
        Set the matrices A, B, Sigma_c, Sigma_r, invokes spectral_radii() to set the spectral radii.
        Args:
            A: np.ndarray, shape: (m, m)
            B: np.ndarray, shape: (n, n)
            Sigma_c: np.ndarray, shape: (n, n)
            Sigma_r: np.ndarray, shape: (m, m)
            normalize: bool, default: False
        Returns:
            None
        """
        self.A = A
        self.B = B
        self.W = np.kron(B, A)
        self.Sigma_c = Sigma_c
        self.Sigma_r = Sigma_r
        self.Sigma_W = np.kron(Sigma_c, Sigma_r)
        self.spectral_radii()  # -> sets rho_A, rho_B, rho_W

        if normalize:
            self.normalize_matrices()

    # —————————————————————————————————————————————————————————————————————————————————————————————————————— #
    def get_matrices(self) -> dict:
        """
        Get the matrices A, B, W, Sigma_c, Sigma_r, Sigma_W.
        """
        return {
            "A": self.A,
            "B": self.B,
            "W": self.W,
            "Sigma_c": self.Sigma_c,
            "Sigma_r": self.Sigma_r,
            "Sigma_W": self.Sigma_W,
        }

    # —————————————————————————————————————————————————————————————————————————————————————————————————————— #
    def normalize_matrices(self) -> None:
        """
        Normalize the matrices A, B, Sigma_c, Sigma_r:
        Rescale matrices using the Frobenius norm, maintaining the kronecker products invariant.
        """
        # Normalize main matrices (A and B)
        F_norm_A = np.linalg.norm(self.A, ord="fro")
        self.A = self.A / F_norm_A
        self.B = self.B * F_norm_A
        # Normalize covariance matrices (Sigma_c and Sigma_r)
        F_norm_Sigma_r = np.linalg.norm(self.Sigma_r, ord="fro")
        self.Sigma_r = self.Sigma_r / F_norm_Sigma_r
        self.Sigma_c = self.Sigma_c * F_norm_Sigma_r
        # Recompute Kronecker product matrix (W)
        self.W = np.kron(self.B, self.A)
        # Recompute Kronecker product covariance matrix (Sigma_W)
        self.Sigma_W = np.kron(self.Sigma_c, self.Sigma_r)

    # —————————————————————————————————————————————————————————————————————————————————————————————————————— #
    def spectral_radii(self) -> float:
        """
        Computes spectral radii of A and B to check the causality condition.
        """
        # Compute Eigenvalues (can be complex)
        eig_vals_A = np.linalg.eigvals(self.A)
        eig_vals_B = np.linalg.eigvals(self.B)
        # Compute Spectral Radii (max absolute value)
        rho_A = np.max(np.abs(eig_vals_A))
        rho_B = np.max(np.abs(eig_vals_B))
        self.rho_A = rho_A
        self.rho_B = rho_B
        self.rho_W = rho_A * rho_B

        return self.rho_A, self.rho_B, self.rho_W

    # —————————————————————————————————————————————————————————————————————————————————————————————————————— #
    def get_spectral_radii(self) -> tuple[float, float, float]:
        """
        Get the spectral radii of A and B.
        """
        return self.rho_A, self.rho_B, self.rho_W

    # —————————————————————————————————————————————————————————————————————————————————————————————————————— #
    def generate_starting_matrices(
        self, X: np.ndarray, return_matrices: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initializes self.A, self.B, self.Sigma_c, and self.Sigma_r using the Projection Estimators (PROJ) method.
        A and B are estimated from the unrestricted VAR(1) matrix Phi_hat.
        Sigma_c and Sigma_r are initialized as identity matrices.
        Args:
            X: np.ndarray, shape: (m, n, T)
            return_matrices: bool, default: False
        Returns:
            tuple[np.ndarray, ..., np.ndarray]: (A, B, W, Sigma_c, Sigma_r, Sigma_W)
        """
        m, n, T = (
            X.shape
        )  # Shape: (m, n, T); m: number of rows, n: number of columns, T: number of time steps
        # ——————————————————————————————————————————————————— #
        # Step 1: Estimate Unrestricted VAR(1) Matrix (Phi_hat)
        X_vec = X.reshape(
            m * n, T, order="F"
        )  # Shape: (mn, T); used Fortran order as per Chen et al. (2020)
        Y = X_vec[:, 1:]  # Columns 1 to T-1 (Target); Shape: (mn, T-1)
        Z = X_vec[:, :-1]  # Columns 0 to T-2 (Predictor); Shape: (mn, T-1)
        # Solve Y = Phi * Z for Phi using Least Squares
        Phi_hat_T, _, _, _ = np.linalg.lstsq(Z.T, Y.T, rcond=None)
        Phi_hat = (
            Phi_hat_T.T
        )  # Shape: (mn, mn); Transposed to match shape A*x = B -> Z.T * Phi.T = Y.T
        # ——————————————————————————————————————————————————— #
        # Step 2: Rearrange Phi_hat into Phi_tilde
        Phi_tilde_cols = []  # Shape: (n^2, m^2)
        for j in range(n):  # Iterate columns of B
            for i in range(n):  # Iterate rows of B
                # Extract the (i, j)-th m x m block from Phi_hat
                row_start, row_end = i * m, (i + 1) * m
                col_start, col_end = j * m, (j + 1) * m
                block = Phi_hat[row_start:row_end, col_start:col_end]
                # Vectorize the block (this is vec(A) scaled by b_{ij})
                Phi_tilde_cols.append(
                    block.flatten(order="F")
                )  # Shape: (m^2, 1) -> Fortran order is mandatory
        Phi_tilde = np.column_stack(Phi_tilde_cols)  # Shape: (m^2, n^2)
        # ——————————————————————————————————————————————————— #
        # Step 3: SVD to find A and B
        U, S, Vt = np.linalg.svd(Phi_tilde, full_matrices=False)
        u1 = U[:, 0]  # First left singular vector (approx vec(A))
        s1 = S[0]  # First singular value
        v1 = Vt[0, :]  # First right singular vector (approx vec(B))
        # Reconstruct A and B -> reshape back to matrices using Fortran order
        self.A = u1.reshape(m, m, order="F")
        self.B = (s1 * v1).reshape(n, n, order="F")
        # ——————————————————————————————————————————————————— #
        # Step 4: Initialize Covariances as identity matrices
        self.Sigma_c = np.eye(n)
        self.Sigma_r = np.eye(m)
        # ——————————————————————————————————————————————————— #
        # Step 5:  Normalize matrices -> updates W and Sigma_W internally
        self.normalize_matrices()
        # ——————————————————————————————————————————————————— #
        # Step 6: Compute spectral radii
        self.spectral_radii()
        if return_matrices:
            return self.A, self.B, self.W, self.Sigma_c, self.Sigma_r, self.Sigma_W

    # —————————————————————————————————————————————————————————————————————————————————————————————————————— #
    def matrices_update(self, X: np.ndarray) -> float:
        """
        Perform one step of the MAR algorithm.
        Args:
            X: np.ndarray, shape: (m, n, T)
        Returns:
            float: log-likelihood
        """
        # ——————————————————————————————————————————————————————————————————————— #

        def log_likelihood(Residuals: np.ndarray) -> float:
            """
            Compute the log-likelihood of the data.
            Args:
                Residuals: np.ndarray, shape: (m, n, T)
            Returns:
                float: log-likelihood
            """
            m, n, T_steps = (
                Residuals.shape
            )  # Shape: (m, n, T); m: number of rows, n: number of columns, T: number of time steps
            # ——————————————————————————————————————————————————— #
            # Term 1: -m * T * log|Sigma_c|
            _, logdet_c = np.linalg.slogdet(self.Sigma_c)  # sign is unused
            term_logdet_c = -m * T_steps * logdet_c
            # ——————————————————————————————————————————————————— #
            # Term 2: -n * T * log|Sigma_r|
            _, logdet_r = np.linalg.slogdet(self.Sigma_r)  # sign is unused
            term_logdet_r = -n * T_steps * logdet_r
            # ——————————————————————————————————————————————————— #
            # Term 3: -tr(Sigma_r^-1 @ R_t @ Sigma_c^-1 @ R_t')
            # re-calculate inverses
            Sigma_r_inv = np.linalg.inv(self.Sigma_r)
            Sigma_c_inv = np.linalg.inv(self.Sigma_c)
            # Compute the trace
            term_trace = -np.einsum(
                "ij, jkt, kl, ilt ->",
                Sigma_r_inv,
                Residuals,
                Sigma_c_inv,
                Residuals,
                optimize=True,
            )  # Einstein summation indices: Sigma_r_inv (ij), Residuals (jkt), Sigma_c_inv (kl), Residuals.T (lit -> ilt)
            # ——————————————————————————————————————————————————— #
            # Compute the log-likelihood
            log_likelihood = term_logdet_c + term_logdet_r + term_trace
            return log_likelihood

        # ——————————————————————————————————————————————————————————————————————— #

        # Define Slices for time summation
        X_t = X[:, :, 1:]  # X_t (current time steps)
        X_tm1 = X[:, :, :-1]  # X_{t-1} (lagged time steps)
        T_steps = X_t.shape[2]  # Number of time steps
        m, n, T = (
            X.shape
        )  # Shape: (m, n, T); m: number of rows, n: number of columns, T: number of time steps
        # Precompute the inverse of Sigma_c and Sigma_r
        Sigma_c_inv = np.linalg.inv(self.Sigma_c)
        Sigma_r_inv = np.linalg.inv(self.Sigma_r)
        # ——————————————————————————————————————————————————— #
        # Update step for A
        # ——————————————————————————————————————————————————— #
        # Precompute the matrices that don't change with t
        mid_Num_A = Sigma_c_inv @ self.B  # For Num_A: Sigma_c^{-1} * B
        mid_Den_A = self.B.T @ Sigma_c_inv @ self.B  # For Den_A: B' * Sigma_c^{-1} * B
        # Vectorized Summation using einsum
        Num_A = np.einsum(
            "mnt, nk, qkt -> mq", X_t, mid_Num_A, X_tm1, optimize=True
        )  # Num_A += X_t * (Sigma_c^{-1} * B) * X_{t-1}'
        Den_A = np.einsum(
            "mnt, nk, qkt -> mq", X_tm1, mid_Den_A, X_tm1, optimize=True
        )  # Den_A += X_{t-1} * (B' * Sigma_c^{-1} * B) * X_{t-1}'
        # Update A <- Num_A * Den_A^{-1}
        self.A = Num_A @ np.linalg.inv(Den_A)
        # ——————————————————————————————————————————————————— #
        # Update step for B
        # ——————————————————————————————————————————————————— #
        # Precompute the matrices that don't change with t
        mid_Num_B = Sigma_r_inv @ self.A  # For Num_B: Sigma_r^{-1} * A
        mid_Den_B = self.A.T @ Sigma_r_inv @ self.A  # For Den_B: A' * Sigma_r^{-1} * A
        # Vectorized Summation using einsum
        Num_B = np.einsum(
            "mnt, mk, kqt -> nq", X_t, mid_Num_B, X_tm1, optimize=True
        )  # Num_B += X_t * (Sigma_r^{-1} * A) * X_{t-1}'
        Den_B = np.einsum(
            "mnt, mk, kqt -> nq", X_tm1, mid_Den_B, X_tm1, optimize=True
        )  # Den_B += X_{t-1} * (A' * Sigma_r^{-1} * A) * X_{t-1}'
        # Update B <- Num_B * Den_B^{-1}
        self.B = Num_B @ np.linalg.inv(Den_B)
        # ——————————————————————————————————————————————————— #
        # Computation of residuals
        # ——————————————————————————————————————————————————— #
        # A is (m, m), X_tm1 is (m, n, T), B is (n, n)
        Residuals = X_t - np.einsum(
            "ij,jkt,lk->ilt", self.A, X_tm1, self.B, optimize=True
        )  # Predictions:  A * X_tm1 * B, Residuals: X_t - Predictions
        # ——————————————————————————————————————————————————— #
        # Update step for Sigma_c
        # ——————————————————————————————————————————————————— #
        Sum_C = np.einsum(
            "mnt, mk, kqt -> nq", Residuals, Sigma_r_inv, Residuals, optimize=True
        )  # Sum_C += R_t' * Sigma_r^{-1} * R_t
        # Update Sigma_c <- (1 / (m * T_steps)) * Sum(R_t' * Sigma_r^{-1} * R_t)
        self.Sigma_c = Sum_C / (m * T_steps)
        # ——————————————————————————————————————————————————— #
        # Update step for Sigma_r
        # ——————————————————————————————————————————————————— #
        # Recompute Sigma_c_inv with the updated Sigma_c
        Sigma_c_inv = np.linalg.inv(self.Sigma_c)
        Sum_R = np.einsum(
            "mnt, nk, qkt -> mq", Residuals, Sigma_c_inv, Residuals, optimize=True
        )  # Sum_R += R_t * Sigma_c^{-1} * R_t'
        # Update Sigma_r <- (1 / (n * T_steps)) * Sum(R_t * Sigma_c^{-1} * R_t')
        self.Sigma_r = Sum_R / (n * T_steps)
        # ——————————————————————————————————————————————————— #
        # Compute log-likelihood before normalization
        ll = log_likelihood(Residuals)
        # ——————————————————————————————————————————————————— #
        # Normalize the matrices -> updates W and Sigma_W internally
        self.normalize_matrices()
        # ——————————————————————————————————————————————————— #
        # Return the log-likelihood
        return ll

    # —————————————————————————————————————————————————————————————————————————————————————————————————————— #
    def fit(
        self,
        X: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6,
        patience: int = 5,
    ) -> tuple[list[float], int, list[float], list[float], list[float]]:
        """
        Fit the model to the data with early stopping.
        Note: Check proper initialization of matrices.
        The data is assumed to be a matrix-valued time series (m, n, T).
        Args:
            X: np.ndarray, shape: (m, n, T)
            max_iter: int, default: 1000
            tol: float, default: 1e-6
            patience: int, default: 5
                Number of consecutive iterations with improvement < tol before stopping
        Returns:
            tuple[list[float], int, list[float], list[float], list[float]]: (log-likelihoods, total_iterations, rho_A_series, rho_B_series, rho_W_series)
        """
        # Initialize variables
        log_likelihoods = []
        rho_A_series = []
        rho_B_series = []
        rho_W_series = []
        total_iterations = max_iter
        no_improvement = 0

        # First iteration out of loop (remove need for "if" statement in loop)
        log_likelihood = self.matrices_update(X)
        log_likelihoods.append(log_likelihood)
        rho_A, rho_B, rho_W = self.spectral_radii()
        rho_A_series.append(rho_A)
        rho_B_series.append(rho_B)
        rho_W_series.append(rho_W)
        for i in range(1, max_iter):
            # Update the matrices
            log_likelihood = self.matrices_update(X)
            log_likelihoods.append(log_likelihood)
            rho_A, rho_B, rho_W = self.spectral_radii()
            rho_A_series.append(rho_A)
            rho_B_series.append(rho_B)
            rho_W_series.append(rho_W)
            # Check for improvement
            improvement = (log_likelihoods[-1] - log_likelihoods[-2]) / np.abs(
                log_likelihoods[-2]
            )
            # Early stopping if no improvement for 'patience' consecutive iterations
            if improvement < tol:
                no_improvement += 1
            else:
                no_improvement = 0  # Reset counter if there's improvement
            if no_improvement >= patience:
                total_iterations = i + 1
                break
        # Return the log-likelihoods, total iterations, and spectral radii
        return (
            log_likelihoods,
            total_iterations,
            rho_A_series,
            rho_B_series,
            rho_W_series,
        )

    # —————————————————————————————————————————————————————————————————————————————————————————————————————— #

    def full_diagnostics(self) -> dict:
        """
        Compute full diagnostics of the model.
        Returns:
            dict: Full diagnostics of the model.
            Keys:
                "eig_gap_A": Eigenvalue gap of A.
                "eig_gap_B": Eigenvalue gap of B.
                "eig_gap_W": Eigenvalue gap of W.
                "eig_ratio_A": Ratio of the first to the second eigenvalue of A.
                "eig_ratio_B": Ratio of the first to the second eigenvalue of B.
                "eig_ratio_W": Ratio of the first to the second eigenvalue of W.
                "trace_A": Trace of A.
                "trace_B": Trace of B.
                "trace_W": Trace of W.
                "cond_A": Condition number of A.
                "cond_B": Condition number of B.
                "cond_W": Condition number of W.
                "noise_scale_ratio": Noise scale ratio (Sigma_c / Sigma_r).
                "log_det_Sigma_c": Log determinant of Sigma_c.
                "log_det_Sigma_r": Log determinant of Sigma_r.
                "log_det_Sigma_W": Log determinant of Sigma_W.
                "Sigma_c_anisotropy": Anisotropy of Sigma_c (max eigenvalue / min eigenvalue).
                "Sigma_r_anisotropy": Anisotropy of Sigma_r (max eigenvalue / min eigenvalue).
                "Sigma_W_anisotropy": Anisotropy of Sigma_W (max eigenvalue / min eigenvalue).
        """

        # Full eigenvalue spectrum
        eig_A = np.linalg.eigvals(self.A)
        eig_B = np.linalg.eigvals(self.B)
        eig_W = np.linalg.eigvals(self.W)
        eig_A_sorted = np.sort(np.abs(eig_A))[::-1]  # Descending order
        eig_B_sorted = np.sort(np.abs(eig_B))[::-1]
        eig_W_sorted = np.sort(np.abs(eig_W))[::-1]

        diagnostics = {
            # Eigenvalue gaps (how dominant is the first eigenvalue?)
            "eig_gap_A": (
                eig_A_sorted[0] - eig_A_sorted[1] if len(eig_A_sorted) > 1 else np.nan
            ),
            "eig_gap_B": (
                eig_B_sorted[0] - eig_B_sorted[1] if len(eig_B_sorted) > 1 else np.nan
            ),
            "eig_gap_W": (
                eig_W_sorted[0] - eig_W_sorted[1] if len(eig_W_sorted) > 1 else np.nan
            ),
            # Ratio: 1st to 2nd eigenvalue (dominance)
            "eig_ratio_A": (
                eig_A_sorted[0] / eig_A_sorted[1] if len(eig_A_sorted) > 1 else np.nan
            ),
            "eig_ratio_B": (
                eig_B_sorted[0] / eig_B_sorted[1] if len(eig_B_sorted) > 1 else np.nan
            ),
            "eig_ratio_W": (
                eig_W_sorted[0] / eig_W_sorted[1] if len(eig_W_sorted) > 1 else np.nan
            ),
            # Sum of all eigenvalues (trace) - shows total "energy"
            "trace_A": np.trace(self.A),
            "trace_B": np.trace(self.B),
            "trace_W": np.trace(self.W),
            # Condition number (largest / smallest singular value)
            "cond_A": np.linalg.cond(self.A),
            "cond_B": np.linalg.cond(self.B),
            "cond_W": np.linalg.cond(self.W),
            # Covariance scale ratio
            "noise_scale_ratio": np.linalg.norm(self.Sigma_c, ord="fro")
            / np.linalg.norm(self.Sigma_r, ord="fro"),
            # Covariance determinants (geometric mean of variances)
            "log_det_Sigma_c": np.linalg.slogdet(self.Sigma_c)[1],
            "log_det_Sigma_r": np.linalg.slogdet(self.Sigma_r)[1],
            "log_det_Sigma_W": np.linalg.slogdet(self.Sigma_W)[1],
            # Covariance anisotropy (directionality of noise)
            "Sigma_c_anisotropy": np.max(np.linalg.eigvals(self.Sigma_c))
            / np.min(np.linalg.eigvals(self.Sigma_c)),
            "Sigma_r_anisotropy": np.max(np.linalg.eigvals(self.Sigma_r))
            / np.min(np.linalg.eigvals(self.Sigma_r)),
            "Sigma_W_anisotropy": np.max(np.linalg.eigvals(self.Sigma_W))
            / np.min(np.linalg.eigvals(self.Sigma_W)),
        }

        return diagnostics


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #
# Multiplex Network class
# —————————————————————————————————————————————————————————————————————————————————————————————————————— #


class Multiplex:
    """
    Multiplex Network class with improved numerical stability.
    Based on Bonaccolto et al. (2019) implementation.
    """

    def __init__(
        self,
        num_layers: int,
        num_assets: int = 28,
        adj_matrices: Optional[list[np.ndarray]] = None,
    ) -> None:
        """
        Initialize the MultiplexNetwork model.
        Args:
            adj_matrices: list[np.ndarray], shape: (m, m), number of layers
            num_assets: int, default: 28
        """
        self.num_layers = num_layers
        self.rho_vector = np.zeros(num_assets)
        self.weights = np.ones(self.num_layers) / self.num_layers
        self.adj_matrices = np.zeros((self.num_layers, num_assets, num_assets))
        if adj_matrices is not None:
            self.set_adjacency_matrices(adj_matrices)

    def get_adjacency_matrices(self) -> list[np.ndarray]:
        """
        Get the adjacency matrices.
        Returns:
            list[np.ndarray]: Adjacency matrices
        """
        return self.adj_matrices

    # —————————————————————————————————————————————————————————————————————————————————————————————————————— #
    def set_adjacency_matrices(
        self, adj_matrices: list[np.ndarray], row_normalize: bool = False
    ) -> Optional[list[np.ndarray]]:
        """
        Update the matrices of the model.
        Preprocesses adjacency matrices: sets diagonal to zero and normalizes by the
        sum of absolute values of each row (row-wise normalization).
        Args:
            adj_matrices: list[np.ndarray], shape: (m, m), number of layers
            scale_B_A: bool, default: False - Scale the matrix B to the matrix A
        """
        preprocessed = []
        for matrix in adj_matrices:
            matrix_copy = matrix.copy()
            # Set diagonal to zero
            np.fill_diagonal(matrix_copy, 0)
            # Row-wise normalization (by sum of absolute values per row)
            if row_normalize:
                row_sums = np.sum(np.abs(matrix_copy), axis=1, keepdims=True)
                # Avoid division by zero for rows with all zeros
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                matrix_normalized = matrix_copy / row_sums
                preprocessed.append(matrix_normalized)
            else:
                eigenvalues = np.linalg.eigvals(matrix_copy)
                spectral_radius = np.max(np.abs(eigenvalues))
                if spectral_radius == 0:
                    matrix_normalized = matrix_copy
                else:
                    matrix_normalized = matrix_copy / spectral_radius
                preprocessed.append(matrix_normalized)
        self.adj_matrices = preprocessed

    # —————————————————————————————————————————————————————————————————————————————————————————————————————— #

    def fit(
        self,
        R: np.ndarray,
        F: np.ndarray,
        max_iter: int = 500,
        ftol: float = 1e-6,
        verbose: bool = False,
    ) -> dict:
        """
        Fit the Multiplex Network model using concentrated maximum likelihood.

        Args:
            R: np.ndarray, shape: (N, T) - DEMEANED Returns matrix
            F: np.ndarray, shape: (T, K) - Factor matrix WITHOUT intercept
            max_iter: int, default: 500 - Maximum optimization iterations
            ftol: float, default: 1e-6 - Function tolerance for convergence
            verbose: bool, default: False - Print optimization progress

        Returns:
            dict: Results containing estimated parameters and diagnostics
        """
        N, T = R.shape
        d = self.num_layers

        # Verify F does not have constant column (common mistake)
        if F.shape[1] > 0:
            F_std = np.std(F, axis=0)
            if np.any(F_std < 1e-10):
                print(
                    "WARNING: Factor matrix F contains near-constant columns. "
                    "Remove intercept - model assumes demeaned returns!"
                )

        # ——————————————————————————————————————————————————————————————————————— #
        # Parameter packing/unpacking helpers
        # θ = [δ₁, ..., δ_d, ρ₁, ..., ρ_N]
        # ——————————————————————————————————————————————————————————————————————— #
        def pack_theta(delta: np.ndarray, rho: np.ndarray) -> np.ndarray:
            return np.concatenate([delta, rho])

        def unpack_theta(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return theta[:d], theta[d:]

        # ——————————————————————————————————————————————————————————————————————— #
        def log_likelihood(theta: np.ndarray) -> float:
            """
            Compute concentrated log-likelihood for given parameters.
            Args:
                theta: np.ndarray, shape: (d + N,)
            Returns:
                float: Log-likelihood
            """
            try:
                delta, rho = unpack_theta(theta)

                # ——————————————————————————————————————————————————— #
                # Step 1: Construct composite network W* = Σ(δ_j · W_j)
                # ——————————————————————————————————————————————————— #
                W_star = np.zeros((N, N))
                for j in range(d):
                    W_star += delta[j] * self.adj_matrices[j]
                # ——————————————————————————————————————————————————— #
                # Step 2: Form spatial filter matrix A = I_N - Ρ·W*
                # ——————————————————————————————————————————————————— #
                Rho = np.diag(rho)
                RhoW = Rho @ W_star
                A = np.eye(N) - RhoW

                # Invertibility / stability check (works with negative weights)
                # A = I - (diag(rho) @ W*). A sufficient condition for invertibility is
                # ||diag(rho) @ W*||_2 < 1 (Neumann series). We use this as a cheap screen;
                # if we're close to the boundary, fall back to the direct eigenvalue check.
                max_rho = np.max(np.abs(rho))
                w_op_norm = np.linalg.norm(W_star, ord=2)  # largest singular value
                if (not np.isfinite(w_op_norm)) or (max_rho * w_op_norm >= 0.999):
                    if verbose:
                        print("WARNING: RhoW is near/over stability boundary")
                    eig_vals = np.linalg.eigvals(RhoW)
                    if np.max(np.abs(eig_vals)) >= 1.0 - 1e-12:
                        if verbose:
                            print("WARNING: RhoW violates |λ| < 1")
                        return -np.inf

                # Check condition number of A
                cond_A = np.linalg.cond(A)
                if cond_A > 1e12:  # Very ill-conditioned
                    print("WARNING: A is likely singular")
                    return -np.inf
                # ——————————————————————————————————————————————————— #
                # Step 3: Compute Jacobian term J = T · log|det(A)|
                # ——————————————————————————————————————————————————— #
                sign, logdet = np.linalg.slogdet(A)
                if sign <= 0:
                    print("WARNING: A is likely singular")
                    return -np.inf
                J = T * logdet
                # ——————————————————————————————————————————————————— #
                # Step 4: Spatial filtering of demeaned returns Z_τ = A · R_τ
                # More efficient: Z = (A @ R).T
                # ——————————————————————————————————————————————————— #
                Z = (A @ R).T  # Shape: [T x N]
                # ——————————————————————————————————————————————————— #
                # Step 5: Concentrated OLS regression
                # ——————————————————————————————————————————————————— #
                # Regress Z on factors F: Z = F·β + E
                # Using lstsq: β = argmin ||Z - F·β||²

                # Method: np.linalg.lstsq (most stable)
                Beta_hat, _, rank, _ = np.linalg.lstsq(
                    F, Z, rcond=None
                )  # Beta_hat shape: [K x N]

                # Check if regression is well-conditioned
                K = F.shape[1]
                if rank < K:
                    print("WARNING: Z is likely rank deficient")
                    return -np.inf  # Rank deficient

                # Compute residuals
                E = Z - F @ Beta_hat  # Shape: [T x N]

                # Compute residual variance
                sigma_sq = np.sum(E**2) / (N * T)
                if sigma_sq <= 0 or not np.isfinite(sigma_sq):
                    print("WARNING: sigma_sq is not finite")
                    return -np.inf

                # ——————————————————————————————————————————————————— #
                # Step 6: log-likelihood ℓ(θ) = J - (NT/2)·log(σ̂²)
                # ——————————————————————————————————————————————————— #
                ll = J - (N * T / 2.0) * np.log(sigma_sq)

                if not np.isfinite(ll):
                    print("WARNING: ll is not finite")
                    return -np.inf

                return ll

            except np.linalg.LinAlgError:
                print("WARNING: LinAlgError in log_likelihood")
                return -np.inf
            except Exception as e:
                if verbose:
                    print(f"Error in log_likelihood: {e}")
                print("WARNING: Exception in log_likelihood")
                return -np.inf

        # ——————————————————————————————————————————————————————————————————————— #
        # Set starting point
        initial_guess = pack_theta(self.weights, self.rho_vector)

        # Bounds: δ_j in [0, 1], ρ_i in [-0.99, 0.99]
        # Note: tighter bounds on ρ to ensure invertibility
        bounds = [(0, 1.0) for _ in range(d)] + [(-0.99, 0.99) for _ in range(N)]

        # Constraint: sum(δ_j) = 1
        constraint_dict = {"type": "eq", "fun": lambda theta: np.sum(theta[:d]) - 1.0}

        # ——————————————————————————————————————————————————— #
        # OPTIMIZATION (minimize negative log-likelihood)
        # ——————————————————————————————————————————————————— #
        result = minimize(
            fun=lambda theta: -log_likelihood(theta),  # Minimize negative
            x0=initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint_dict,
            options={"maxiter": max_iter, "ftol": ftol, "disp": verbose},
        )

        # ——————————————————————————————————————————————————— #
        # EXTRACT RESULTS AND UPDATE MODEL
        # ——————————————————————————————————————————————————— #
        theta_hat = result.x
        delta_hat, rho_hat = unpack_theta(theta_hat)

        # Update model parameters
        self.rho_vector = rho_hat
        self.weights = delta_hat

        # Construct estimated composite network Ŵ*
        W_star_hat = np.zeros((N, N))
        for j in range(d):
            W_star_hat += delta_hat[j] * self.adj_matrices[j]

        # Construct estimated spatial filter Â = I_N - Ρ̂·Ŵ*
        Rho_hat = np.diag(rho_hat)
        A_hat = np.eye(N) - Rho_hat @ W_star_hat
        # Compute fitted values using a stable least squares solver
        Z_hat = (A_hat @ R).T  # Shape: [T x N]
        Beta_hat, _, _, _ = np.linalg.lstsq(
            F, Z_hat, rcond=None
        )  # Compute Beta_hat using lstsq (NO intercept)
        # Residuals in filtered space
        E_hat = Z_hat - F @ Beta_hat
        # Final residual variance
        sigma_sq_hat = np.sum(E_hat**2) / (N * T)
        # Final log-likelihood
        _, logdet = np.linalg.slogdet(A_hat)
        log_lik_final = T * logdet - (N * T / 2.0) * np.log(sigma_sq_hat)

        # ——————————————————————————————————————————————————— #
        # Return results dictionary
        # ——————————————————————————————————————————————————— #
        return {
            "delta_hat": delta_hat,
            "rho_hat": rho_hat,
            "beta_hat": Beta_hat,
            "W_star": W_star_hat,
            "A_hat": A_hat,
            "sigma_sq": sigma_sq_hat,
            "log_likelihood": log_lik_final,
            "success": result.success,
            "message": result.message,
            "nit": result.nit,
            "condition_number_A": np.linalg.cond(A_hat),
        }


# —————————————————————————————————————————————————————————————————————————————————————————————————————— #
