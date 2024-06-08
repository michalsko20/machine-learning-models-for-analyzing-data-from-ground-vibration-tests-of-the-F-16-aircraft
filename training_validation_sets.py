import numpy as np
from sklearn.model_selection import train_test_split
from samples import DataSample
from numpy.typing import NDArray


def get_training_validation_and_testing_sets(
    data_samples: list[DataSample], window_size: int, scaled: bool = False
) -> tuple[NDArray]:
    X, y = [], []
    sets_size = int(len(data_samples) / 2)

    for i in range(0, sets_size, window_size):
        window = data_samples[i : i + window_size]
        X_window = np.array([sample.X for sample in window])
        y_window = np.array(window[-1].y)
        X.append(X_window)
        y.append(y_window)

    X = np.array(X, dtype="object")
    y = np.array(y, dtype="object")

    if scaled:
        X = _get_scaled_X(X)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


def _get_scaled_X(X: NDArray) -> NDArray:
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    numerical_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )

    X_reshaped = np.array(X, dtype=object).reshape(-1, 1)
    X_transformed = numerical_pipeline.fit_transform(X_reshaped)
    X_transformed_reshaped = X_transformed.reshape(-1, 500, 2)

    return X_transformed_reshaped
