import numpy as np


def compute_confidence(impacts, ekf_cov_trace=None):
    """
    Computes confidence score (0-100) based on Monte Carlo spread
    and optional EKF uncertainty.

    Spread is normalized against a max reference distance (5000m),
    so confidence varies meaningfully rather than collapsing to 0 or 100.
    """

    if impacts is None or len(impacts) < 10:
        return 0.0

    impacts = np.asarray(impacts)

    center = np.mean(impacts, axis=0)

    distances = np.linalg.norm(impacts - center, axis=1)

    spread = np.std(distances)

    # Normalize: 0m spread = 100% confidence, 5000m spread = 0% confidence
    max_spread = 5000.0
    confidence = max(0.0, 1.0 - (spread / max_spread))

    # Optionally reduce confidence based on EKF covariance trace
    if ekf_cov_trace is not None:
        confidence *= np.exp(-ekf_cov_trace * 0.001)

    return float(np.clip(confidence * 100, 0, 100))