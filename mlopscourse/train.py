import os

import fire
import mlflow
from hydra import compose
from omegaconf import DictConfig, OmegaConf

from .data.prepare_dataset import load_dataset
from .models.models_zoo import prepare_model
from .utils import get_git_revision_hash


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
        self.cfg.logging.commit_id = get_git_revision_hash()
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
        print("The training was finished successfully!\nCollecting logs...")

        # Since there is no easy way to log metrics as functions of time during
        # the training, they should be collected after it.
        mlflow_cfg = self.cfg.logging.mlflow
        mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
        exp_id = mlflow.set_experiment(mlflow_cfg.exp_name).experiment_id
        with mlflow.start_run(
            experiment_id=exp_id, run_name=f"training-{self.cfg.model.name}"
        ):
            signature = mlflow.models.infer_signature(X_train, y_train)
            # Unfortunately, logging is model dependent, at least because
            # RandomForestRegressor doesn't provide the target metric progress.
            if self.cfg.model.name == "cb":
                mlflow.catboost.save_model(
                    model.model,
                    f"checkpoints/mlflow_{self.cfg.model.name}_ckpt/",
                    signature=signature,
                )
                model.log_fis_and_metrics(exp_id, X_train.columns)
            else:
                mlflow.sklearn.save_model(
                    model.model,
                    f"checkpoints/mlflow_{self.cfg.model.name}_ckpt/",
                    signature=signature,
                )
                model.log_fis_and_metrics(exp_id, X_train, y_train)


if __name__ == "__main__":
    fire.Fire(Trainer)
