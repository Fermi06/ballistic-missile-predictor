# radar/noise.py

import numpy as np


def add_measurement_noise(z):

    noisy = np.copy(z)

    noisy[0] += np.random.normal(
        0,
        25.0
    )

    noisy[1] += np.radians(
        np.random.normal(0, 0.05)
    )

    noisy[2] += np.radians(
        np.random.normal(0, 0.05)
    )

    return noisy