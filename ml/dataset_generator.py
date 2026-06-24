# ml/dataset_generator.py

import numpy as np
import pandas as pd

from simulation.missile import Missile
from simulation.integrator import propagate


class DatasetGenerator:

    def __init__(
        self,
        samples=5000,
        observation_steps=10
    ):
        self.samples = samples
        self.observation_steps = observation_steps

    def generate(self):

        rows = []

        for idx in range(self.samples):

            vx = np.random.uniform(
                400,
                1200
            )

            vy = np.random.uniform(
                -200,
                200
            )

            vz = np.random.uniform(
                400,
                1200
            )

            missile = Missile()

            initial_state = [
                0.0,
                0.0,
                0.0,
                vx,
                vy,
                vz
            ]

            try:

                sol = propagate(
                    missile,
                    initial_state,
                    duration=300
                )

            except Exception:
                continue

            if len(sol.t) < (
                self.observation_steps + 1
            ):
                continue

            impact_x = float(
                sol.y[0][-1]
            )

            impact_y = float(
                sol.y[1][-1]
            )

            row = {}

            for i in range(
                self.observation_steps
            ):

                row[f"x_{i}"] = float(
                    sol.y[0][i]
                )

                row[f"y_{i}"] = float(
                    sol.y[1][i]
                )

                row[f"z_{i}"] = float(
                    sol.y[2][i]
                )

                row[f"vx_{i}"] = float(
                    sol.y[3][i]
                )

                row[f"vy_{i}"] = float(
                    sol.y[4][i]
                )

                row[f"vz_{i}"] = float(
                    sol.y[5][i]
                )

            row["impact_x"] = impact_x
            row["impact_y"] = impact_y

            rows.append(row)

            if (idx + 1) % 100 == 0:

                print(
                    f"Generated "
                    f"{idx + 1}/"
                    f"{self.samples}"
                )

        return pd.DataFrame(rows)


if __name__ == "__main__":

    generator = DatasetGenerator(
        samples=2000,
        observation_steps=10
    )

    df = generator.generate()

    print(df.head())

    print(
        "Dataset shape:",
        df.shape
    )

    df.to_csv(
        "missile_dataset.csv",
        index=False
    )

    print(
        "Saved missile_dataset.csv"
    )