# tracking/ekf.py

import numpy as np


class ExtendedKalmanFilter:

    def __init__(self):

        # State:
        # [x, y, z, vx, vy, vz]

        self.x = np.zeros(6)

        self.P = np.eye(6) * 1000.0

        # Process noise
        self.Q = np.diag([
            1.0,
            1.0,
            1.0,
            10.0,
            10.0,
            10.0
        ])

        # Radar measurement noise
        self.R = np.diag([
            25.0**2,
            np.radians(0.05)**2,
            np.radians(0.05)**2
        ])

    # -------------------------
    # State Prediction
    # -------------------------

    def predict(self, dt):

        F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])

        self.x = F @ self.x

        self.P = (
            F @ self.P @ F.T
            + self.Q
        )

    # -------------------------
    # Radar Measurement Model
    # -------------------------

    def h(self, state):

        px, py, pz = state[:3]

        r = np.sqrt(
            px**2 +
            py**2 +
            pz**2
        )

        if r < 1e-6:
            r = 1e-6

        azimuth = np.arctan2(
            py,
            px
        )

        elevation = np.arcsin(
            np.clip(
                pz / r,
                -1.0,
                1.0
            )
        )

        return np.array([
            r,
            azimuth,
            elevation
        ])

    # -------------------------
    # Numerical Jacobian
    # -------------------------

    def jacobian(self, state):

        eps = 1e-5

        H = np.zeros((3, 6))

        h0 = self.h(state)

        for i in range(6):

            perturbed = state.copy()

            perturbed[i] += eps

            H[:, i] = (
                self.h(perturbed)
                - h0
            ) / eps

        return H

    # -------------------------
    # Angle Wrapping
    # -------------------------

    def normalize_angle(self, angle):

        return (
            angle + np.pi
        ) % (2 * np.pi) - np.pi

    # -------------------------
    # Measurement Update
    # -------------------------

    def update(self, measurement):

        H = self.jacobian(
            self.x
        )

        predicted_measurement = (
            self.h(self.x)
        )

        innovation = (
            measurement
            - predicted_measurement
        )

        innovation[1] = (
            self.normalize_angle(
                innovation[1]
            )
        )

        innovation[2] = (
            self.normalize_angle(
                innovation[2]
            )
        )

        S = (
            H @ self.P @ H.T
            + self.R
        )

        K = (
            self.P
            @ H.T
            @ np.linalg.inv(S)
        )

        self.x = (
            self.x
            + K @ innovation
        )

        I = np.eye(
            self.P.shape[0]
        )

        self.P = ((I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T)

    # -------------------------
    # Convenience Methods
    # -------------------------

    def get_position(self):

        return self.x[:3]

    def get_velocity(self):

        return self.x[3:]

    def get_state(self):

        return self.x.copy()