import numpy as np


def wind_velocity(altitude: float):
    if altitude < 5000:
        return np.array([5.0, 0.0, 0.0])

    elif altitude < 15000:
        return np.array([15.0, 0.0, 0.0])

    else:
        return np.array([30.0, 0.0, 0.0])