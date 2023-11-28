import fire
from hydra import initialize

from mlopscourse.infer import Inferencer
from mlopscourse.train import Trainer


def train(
    config_name: str,
    config_path: str = "configs/",
    hydra_version_base: str = "1.3",
    **kwargs: dict,
) -> None:
    """
    Trains the chosen model on the train split of the dataset and saves the checkpoint.

    Parameters
    ----------
    config_name : str
        The name of the configuration file to use for model, training and inference
        hyperparameters.
    config_path : str
        The path to the configuration files.
    hydra_version_base : str
        The compatibility level of hydra to use.
    **kwargs : dict, optional
        Values of the configuration file to override.
    """
    with initialize(config_path=config_path, version_base=hydra_version_base):
        Trainer(config_name, **kwargs).train()


def infer(
    config_name: str,
    config_path: str = "configs/",
    hydra_version_base: str = "1.3",
    **kwargs: dict,
) -> None:
    """
    Runs the chosen model on the test set of the dataset and calculates the R^2 metric.

    Parameters
    ----------
    config_name : str
        The name of the configuration file to use for model, training and inference
        hyperparameters.
    config_path : str
        The path to the configuration files.
    hydra_version_base : str
        The compatibility level of hydra to use.
    **kwargs : dict, optional
        Values of the configuration file to override.
    """
    with initialize(config_path=config_path, version_base=hydra_version_base):
        Inferencer(config_name, **kwargs).infer()


if __name__ == "__main__":
    fire.Fire()
