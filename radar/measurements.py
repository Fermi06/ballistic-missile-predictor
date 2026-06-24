import numpy as np

def cartesian_to_radar(state):

    x, y, z = state[:3]

    r = np.sqrt(
        x**2 +
        y**2 +
        z**2
    )

    azimuth = np.arctan2(y, x)

    elevation = np.arcsin(
        z / r
    )

    return np.array([
        r,
        azimuth,
        elevation
    ])