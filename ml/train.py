# ml/train.py

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.multioutput import (
    MultiOutputRegressor
)

from ml.models import (
    random_forest
)


def train():

    df = pd.read_csv(
        "missile_dataset.csv"
    )

    X = df.drop(
        columns=[
            "impact_x",
            "impact_y"
        ]
    )

    y = df[
        [
            "impact_x",
            "impact_y"
        ]
    ]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    model = MultiOutputRegressor(
        random_forest()
    )

    print("Training...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"MAE  : {mae:.2f}"
    )

    print(
        f"RMSE : {rmse:.2f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    joblib.dump(
        model,
        "impact_predictor.pkl"
    )

    print(
        "Saved impact_predictor.pkl"
    )


if __name__ == "__main__":
    import numpy as np
    train()