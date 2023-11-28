import os
import pickle

import fire
from hydra import compose
from omegaconf import DictConfig, OmegaConf

from .data.prepare_dataset import load_dataset


class Inferencer:
    """
    Runs the chosen model on the test set of the dataset and calculates the R^2 metric.

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

    def infer(self) -> None:
        (
            X_test,
            y_test,
            _,
            _,
        ) = load_dataset(split="test")

        with open(f"checkpoints/{self.cfg.inference.checkpoint_name}", "rb") as f:
            model = pickle.load(f)
        print(f"Evaluating the {self.cfg.model.name} model...")
        y_preds = model.eval(X_test, y_test)

        os.makedirs("predictions", exist_ok=True)
        ckpt_name = self.cfg.inference.checkpoint_name.split(".")[0]
        y_preds.to_csv(f"predictions/{ckpt_name}_preds.csv")


if __name__ == "__main__":
    fire.Fire(Inferencer)
