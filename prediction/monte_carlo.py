import numpy as np
from prediction.impact_predictor import ImpactPredictor


class MonteCarloImpact:

    def __init__(self, samples=1000):
        self.samples = samples
        self.predictor = ImpactPredictor()

    def _sanitize_covariance(self, cov):
        """
        Fix covariance matrix so it is safe for sampling.
        """

        cov = np.asarray(cov)

        # Remove NaN / Inf
        cov = np.nan_to_num(cov)

        # Force symmetry
        cov = 0.5 * (cov + cov.T)

        # Eigenvalue repair (make PSD)
        eigvals, eigvecs = np.linalg.eigh(cov)

        eigvals = np.clip(eigvals, 1e-10, None)

        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Extra safety jitter
        cov += 1e-8 * np.eye(cov.shape[0])

        return cov

    def run(self, state, covariance):

        impacts = []

        # Validate state
        state = np.asarray(state)
        state = np.nan_to_num(state)

        covariance = self._sanitize_covariance(covariance)

        dim = len(state)

        # Ensure correct shape
        if covariance.shape != (dim, dim):
            raise ValueError(
                f"Covariance shape {covariance.shape} "
                f"does not match state dimension {dim}"
            )

        for _ in range(self.samples):

            sample = np.random.multivariate_normal(
                mean=state,
                cov=covariance
            )

            # Safety check
            if np.any(np.isnan(sample)):
                continue

            result = self.predictor.predict(sample)

            # Validate output
            if result is None:
                continue

            ix = result.get("impact_x", np.nan)
            iy = result.get("impact_y", np.nan)

            if np.isnan(ix) or np.isnan(iy):
                continue

            impacts.append([ix, iy])

        return np.array(impacts)

    def confidence_radius(self, impacts):

        if len(impacts) == 0:
            return 0.0

        center = np.mean(impacts, axis=0)

        distances = np.linalg.norm(
            impacts - center,
            axis=1
        )

        return np.percentile(distances, 95)