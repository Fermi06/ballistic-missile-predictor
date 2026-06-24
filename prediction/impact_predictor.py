# prediction/impact_predictor.py

import numpy as np

G = 9.81


class ImpactPredictor:

    def __init__(
        self,
        dt=0.1,
        max_time=1000
    ):
        self.dt = dt
        self.max_time = max_time

    def predict(self, state):

        x, y, z, vx, vy, vz = state

        t = 0.0

        while (
            z > 0 and
            t < self.max_time
        ):

            x += vx * self.dt
            y += vy * self.dt

            z += vz * self.dt

            vz -= (
                G * self.dt
            )

            t += self.dt

        return {
            "impact_x": x,
            "impact_y": y,
            "time_to_impact": t
        }