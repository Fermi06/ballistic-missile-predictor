# ml/models.py

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


def random_forest():

    return RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )


def gradient_boosting():

    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )