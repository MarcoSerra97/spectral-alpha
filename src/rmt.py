import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MarchenkoPasturFilter(BaseEstimator, TransformerMixin):
    """
    Filters the sample correlation matrix using the Marchenko-Pastur law.

    The Marchenko-Pastur law gives the exact theoretical distribution of
    eigenvalues of a random correlation matrix with N assets and T
    observations, in the limit N, T -> infinity with q = N/T fixed.

    Any eigenvalue above the upper edge lambda_+ is considered to carry
    genuine statistical information (a 'signal' eigenvalue). Eigenvalues
    below lambda_+ are noise and are replaced by their theoretical mean,
    which preserves the trace of the matrix.

    Parameters
    ----------
    variance : float
        Assumed variance of the random entries. For a correlation matrix
        this is 1.0 by default.

    Attributes
    ----------
    n_assets_ : int
        Number of assets N seen during fit.
    n_obs_ : int
        Number of observations T seen during fit.
    q_ : float
        Ratio N/T seen during fit.
    lambda_plus_ : float
        MP upper edge computed during fit.
    n_signal_ : int
        Number of eigenvalues above lambda_plus_ (signal eigenvalues).
    correlation_matrix_ : np.ndarray
        The MP-filtered correlation matrix computed during fit.
    eigenvalues_ : np.ndarray
        Raw eigenvalues of the sample correlation matrix.
    eigenvalues_cleaned_ : np.ndarray
        Eigenvalues after MP filtering.
    """

    def __init__(self, variance: float = 1.0):
        self.variance = variance

    def fit(self, X: np.ndarray, y=None) -> "MarchenkoPasturFilter":
        """
        Compute the MP-filtered correlation matrix from the returns matrix X.

        Parameters
        ----------
        X : np.ndarray of shape (T, N)
            Matrix of returns. Rows are time observations, columns are assets.
        """
        X = np.array(X)
        T, N = X.shape

        self.n_obs_ = T
        self.n_assets_ = N
        self.q_ = N / T

        # MP upper edge — the exact theoretical bound
        # any eigenvalue above this carries genuine information
        self.lambda_plus_ = self.variance * (1 + np.sqrt(self.q_)) ** 2

        #This is the exact theoretical result from random matrix theory. For a matrix of IID random entries with variance σ2\sigma^2
        # q=N/T = N/T the largest eigenvalue of the sample correlation matrix converges to  \lambda_+=\sigma^2(1+\sqrt{q})^2.
        #Any empirical eigenvalue above this cannot be explained by random chance — it carries genuine statistical signal.
        #Any eigenvalue below or equal to \lambda_+ is statistically indistinguishable from what we would get if returns were pure i.i.d with 
        #no correlation at all
        

        # compute sample correlation matrix
        C = np.corrcoef(X.T)
        #np.corrcoef computes the Pearson correlation matrix. 
        #It expects input of shape (N, T) — variables as rows, observations as columns
        #  — which is why you transpose X with .T. X is (T, N) so X.T is (N, T). 
        # The result C is an (N, N) symmetric matrix with ones on the diagonal.
        #General formula for Pearson correlation matrix C of a matrix C is C_{ij}=Cov(X_i,X_j)/\sigma_i \sigma_k 
        # where sigma's are std and Cov is the covariance matrix between asset i-th and asset j-th, which are the columns of the matrix X
        # In matrix form C= D^{-1}\Sigma D^{-1} where \Sigma=cov matrix and D=diag(\sigma_1,\dots,\sigma_N)
        # # manual version — same result as np.corrcoef(X.T)
        #X_centered = X - X.mean(axis=0)
        #X_standardized = X_centered / X.std(axis=0)
        #C = (X_standardized.T @ X_standardized) / (T - 1) 


        # eigendecomposition — eigh is faster and more stable than eig
        # for symmetric matrices and guarantees real eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        #eigh is specifically for symmetry (Hermitian) matrices, hence faster than the more generic eig.
        #Returns eigenvalues sorted in ascending order, smaller first largest last.
        #eignevectors is an (N,N) matrix where each columns is an eigenvector corresponding to the eigenvalue at the same position

        self.eigenvalues_ = eigenvalues.copy()
        #Saves a copy of the original eigenvalues before modification. 
        # .copy() is important — without it self.eigenvalues_ would be a reference to the same array, 
        # and modifying eigenvalues later would silently change self.eigenvalues_ too.


        # count signal eigenvalues
        self.n_signal_ = int(np.sum(eigenvalues > self.lambda_plus_))
        #eigenvalues > self.lambda_plus_ produces a boolean array — True where an eigenvalue exceeds λ+\lambda_+
        #  False otherwise. np.sum() counts the True values (treating True as 1 and False as 0). 
        # int() converts from numpy integer to plain Python integer. 
        # This tells you how many genuine signal components the correlation matrix contains.


        # replace noise eigenvalues with their theoretical mean
        # the mean of the MP distribution is variance * 1 = 1.0
        # this preserves the trace of the matrix
        noise_mean = np.mean(
            eigenvalues[eigenvalues <= self.lambda_plus_]
        )
        eigenvalues_cleaned = np.where(
            eigenvalues > self.lambda_plus_,
            eigenvalues,
            noise_mean,
        )
     #Takes only the noise eigenvalues — those at or below λ+ and computes their mean. 
     # This is the value that will replace each noise eigenvalue. 
     # Using the mean rather than zero preserves the trace of the matrix 
     # (the sum of eigenvalues equals the sum of diagonal entries of C, which is N for a correlation matrix). 
     # Setting noise eigenvalues to zero would shrink the trace and distort the matrix.

        self.eigenvalues_cleaned_ = eigenvalues_cleaned

        # reconstruct cleaned correlation matrix
        C_cleaned = eigenvectors @ np.diag(eigenvalues_cleaned) @ eigenvectors.T

        # renormalize diagonal to 1 to restore valid correlation matrix
        D = np.sqrt(np.diag(C_cleaned))
        C_cleaned = C_cleaned / np.outer(D, D)

        # clip to [-1, 1] to correct for floating point errors
        self.correlation_matrix_ = np.clip(C_cleaned, -1.0, 1.0)

        #this result is stored as ther final cleaned correlation matrix

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Project the returns matrix through the inverse of the cleaned
        correlation matrix. This decorrelates the features using the
        noise-filtered covariance structure.

        Parameters
        ----------
        X : np.ndarray of shape (T, N)
            Matrix of returns.

        Returns
        -------
        np.ndarray of shape (T, N)
            Decorrelated returns.
        """
        C_inv = np.linalg.pinv(self.correlation_matrix_)
        return X @ C_inv
     #np.linalg.pinv computes the Moore-Penrose pseudoinverse of the cleaned correlation matrix. 
     # The pseudoinverse is used rather than the regular inverse because the cleaned matrix may be singular or near-singular
     #  — the pseudoinverse handles this gracefully. 
     # X @ C_inv projects the returns matrix through the inverse correlation structure, which decorrelates the features. 
     # The result has the same shape as X.



    def get_lambda_plus(self) -> float:
        """Return the MP upper edge."""
        return self.lambda_plus_
    #simpler getter for \lambda_+

    def summary(self) -> dict:
        """
        Return a summary of the filter results.
        Useful for logging and the research notebook.
        """
        return {
            "n_assets": self.n_assets_,
            "n_obs": self.n_obs_,
            "q": round(self.q_, 4),
            "lambda_plus": round(self.lambda_plus_, 4),
            "n_signal_eigenvalues": self.n_signal_,
            "n_noise_eigenvalues": self.n_assets_ - self.n_signal_,
            "largest_eigenvalue": round(self.eigenvalues_[-1], 4),
            "second_eigenvalue": round(self.eigenvalues_[-2], 4),
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data import load_data

    print("Loading returns...")
    returns = load_data()
    #loads the full returns DataFrame

    # We then use as single training window the first 5 years (approx 260 weeks)
    # rows = time observations, columns = assets
    window = returns.iloc[:260].dropna(axis=1) #we need to drop any columns that have NaN values in this window, because np.corrcoef cannot handel NaN
    print(f"Training window shape: {window.shape} (T={window.shape[0]}, N={window.shape[1]})")

    # fit the filter
    rmt = MarchenkoPasturFilter()
    rmt.fit(window.values)
 
 # window is a DataFrame — shape (260, Nassets)
# it looks like this:
#
#             AAPL    MSFT    GOOG  ...
# 2005-01-14  0.012  -0.003   0.007
# 2005-01-21 -0.005   0.011  -0.002
# ...

#window.values
# returns a plain numpy array — shape (260, Nassets)
# array([[ 0.012, -0.003,  0.007, ...],
#        [-0.005,  0.011, -0.002, ...],
#        ...])
 

    # print summary
    summary = rmt.summary()
    print("\nMP Filter summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nSanity checks:")
    print(f"  Diagonal of cleaned correlation matrix (should be ~1.0):")
    diag = np.diag(rmt.correlation_matrix_)
    print(f"    min={diag.min():.6f}, max={diag.max():.6f}, mean={diag.mean():.6f}")
    print(f"  Eigenvalues range: [{rmt.eigenvalues_.min():.4f}, {rmt.eigenvalues_.max():.4f}]")
    print(f"  Cleaned eigenvalues range: [{rmt.eigenvalues_cleaned_.min():.4f}, {rmt.eigenvalues_cleaned_.max():.4f}]")
    print(f"  lambda_plus = {rmt.lambda_plus_:.4f}")
    print(f"  Eigenvalues above lambda_plus: {rmt.n_signal_}")