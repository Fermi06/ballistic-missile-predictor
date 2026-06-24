import numpy as np

from simulation.atmosphere import air_density
from simulation.wind import wind_velocity


G = 9.81


class Missile:

    def __init__(
        self,
        mass=1000.0,
        drag_coefficient=0.25,
        cross_section=0.3,
    ):
        self.mass = mass
        self.cd = drag_coefficient
        self.area = cross_section

    def derivatives(self, t, state):

        x, y, z, vx, vy, vz = state

        rho = air_density(z)

        wind = wind_velocity(z)

        relative_velocity = np.array(
            [vx, vy, vz]
        ) - wind

        speed = np.linalg.norm(
            relative_velocity
        )

        drag_force = (
            0.5
            * rho
            * self.cd
            * self.area
            * speed**2
        )

        if speed > 1e-6:
            drag_direction = (
                relative_velocity
                / speed
            )
        else:
            drag_direction = np.zeros(3)

        drag_acceleration = (
            -drag_force
            * drag_direction
            / self.mass
        )

        gravity = np.array(
            [0.0, 0.0, -G]
        )

        acceleration = (
            drag_acceleration
            + gravity
        )

        return np.array([
            vx,
            vy,
            vz,
            acceleration[0],
            acceleration[1],
            acceleration[2],
        ])