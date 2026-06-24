# tracking/fusion.py

import numpy as np


def fuse_estimates(estimates, covariances):
    """
    Fuse multiple state estimates using covariance-weighted averaging
    (simplified Kalman fusion / Bar-Shalom–Campo formula).

    Parameters
    ----------
    estimates   : list of np.ndarray, each shape (n,)
    covariances : list of np.ndarray, each shape (n, n)

    Returns
    -------
    fused_state : np.ndarray, shape (n,)
    fused_cov   : np.ndarray, shape (n, n)
    """

    if len(estimates) == 0:
        raise ValueError("No estimates provided to fuse.")

    if len(estimates) == 1:
        return estimates[0].copy(), covariances[0].copy()

    # Compute inverse-covariance-weighted sum
    n = estimates[0].shape[0]
    info_sum = np.zeros((n, n))
    state_sum = np.zeros(n)

    for x, P in zip(estimates, covariances):
        P_inv = np.linalg.inv(P)
        info_sum += P_inv
        state_sum += P_inv @ x

    fused_cov = np.linalg.inv(info_sum)
    fused_state = fused_cov @ state_sum

    return fused_state, fused_cov