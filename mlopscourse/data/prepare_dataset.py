from typing import List, Tuple

import fire
import pandas as pd
from sklearn.datasets import fetch_openml


def prepare_dataset(print_info: bool = True) -> None:
    bikes = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas")
    # Make an explicit copy to avoid "SettingWithCopyWarning" from pandas
    X, y = bikes.data.copy(), bikes.target

    # Because of this rare category, we collapse it into "rain".
    X["weather"].replace(to_replace="heavy_rain", value="rain", inplace=True)

    # We can see that we have data from two years. We use the first year
    # to train the model and the second year to test the model.
    mask_training = X["year"] == 0
    X = X.drop(columns=["year"])
    X_train, y_train = X[mask_training], y[mask_training]
    X_test, y_test = X[~mask_training], y[~mask_training]
    if print_info:
        X_train.info()

    X_train = X_train.assign(bikes=y_train.values)
    X_train.to_csv("mlopscourse/data/train_split.csv")
    X_test = X_test.assign(bikes=y_test.values)
    X_test.to_csv("mlopscourse/data/test_split.csv")


def load_dataset(split: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    X = pd.read_csv(f"mlopscourse/data/{split}_split.csv", index_col=0)
    y = X["bikes"]
    X = X.drop(columns=["bikes"])

    numerical_features = [
        "temp",
        "feel_temp",
        "humidity",
        "windspeed",
    ]
    categorical_features = X.columns.drop(numerical_features).values.tolist()

    return (X, y, numerical_features, categorical_features)


if __name__ == "__main__":
    fire.Fire(prepare_dataset)
