import os

import fire

from data.prepare_dataset import prepare_dataset
from models.models_zoo import prepare_model


def train(model_type: str) -> None:
    """
    Trains the chosen model on the train split of the dataset and saves the checkpoint.

    Parameters
    ----------
    model_type : str
        The type of model for training. Should be "rf" for RandomForest and "cb"
        for CatBoost.
    """
    (
        X_train,
        y_train,
        _,
        _,
        numerical_features,
        categorical_features,
    ) = prepare_dataset()
    model = prepare_model(model_type, numerical_features, categorical_features)

    print(f"Training the {model_type} model...")
    model.train(X_train, y_train)

    os.makedirs("checkpoints", exist_ok=True)
    model.save_checkpoint("checkpoints/")
    print("The training was finished successfully!")


if __name__ == "__main__":
    fire.Fire(train)
