import os

import fire
from hydra import compose
from omegaconf import DictConfig, OmegaConf

from .data.prepare_dataset import load_dataset
from .models.models_zoo import prepare_model


class Trainer:
    """
    Trains the chosen model on the train split of the dataset and saves the checkpoint.

    Attributes
    ----------
    cfg : omegaconf.DictConfig
        The configuration containing the model type and hyperparameters, training and
        inference parameters.
    """

    def __init__(self, config_name: str, **kwargs: dict) -> None:
        self.cfg: DictConfig = compose(
            config_name=config_name, overrides=[f"{k}={v}" for k, v in kwargs.items()]
        )
        print(OmegaConf.to_yaml(self.cfg))

    def train(self) -> None:
        (
            X_train,
            y_train,
            numerical_features,
            categorical_features,
        ) = load_dataset(split="train")

        model = prepare_model(self.cfg, numerical_features, categorical_features)

        print(f"Training the {self.cfg.model.name} model...")
        model.train(X_train, y_train)

        os.makedirs("checkpoints", exist_ok=True)
        model.save_checkpoint("checkpoints/")
        print("The training was finished successfully!")


if __name__ == "__main__":
    fire.Fire(Trainer)
