import os

import fire

from .data.prepare_dataset import load_dataset
from .models.models_zoo import prepare_model


class Trainer:
    """
    Trains the chosen model on the train split of the dataset and saves the checkpoint.

    Attributes
    ----------
    model_type : str
        The type of model for training. Should be "rf" for RandomForest and "cb"
        for CatBoost.
    """

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type

    def train(self) -> None:
        (
            X_train,
            y_train,
            numerical_features,
            categorical_features,
        ) = load_dataset(split="train")
        model = prepare_model(self.model_type, numerical_features, categorical_features)

        print(f"Training the {self.model_type} model...")
        model.train(X_train, y_train)

        os.makedirs("checkpoints", exist_ok=True)
        model.save_checkpoint("checkpoints/")
        print("The training was finished successfully!")


if __name__ == "__main__":
    fire.Fire(Trainer)
