# simulation/atmosphere.py

import numpy as np

SEA_LEVEL_DENSITY = 1.225
SCALE_HEIGHT = 8500.0


def air_density(altitude: float) -> float:
   
    altitude = max(0.0, altitude)

    return SEA_LEVEL_DENSITY * np.exp(
        -altitude / SCALE_HEIGHT
    )