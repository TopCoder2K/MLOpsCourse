from sklearn.datasets import fetch_openml


def prepare_dataset() -> tuple:
    bikes = fetch_openml(
        "Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas"
    )
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
    X_train.info()

    numerical_features = [
        "temp",
        "feel_temp",
        "humidity",
        "windspeed",
    ]
    categorical_features = X_train.columns.drop(numerical_features)

    return (
        X_train, y_train, X_test, y_test,
        numerical_features, categorical_features
    )
