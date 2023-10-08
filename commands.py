import fire

from mlopscourse.infer import Inferencer
from mlopscourse.train import Trainer


def train(model_type: str) -> None:
    """
    Trains the chosen model on the train split of the dataset and saves the checkpoint.

    Parameters
    ----------
    model_type : str
        The type of model for training. Should be "rf" for RandomForest and "cb"
        for CatBoost.
    """
    Trainer(model_type).train()


def infer(model_type: str, ckpt: str) -> None:
    """
    Runs the chosen model on the test set of the dataset and calculates the R^2 metric.

    Parameters
    ----------
    model_type : str
        The type of model that was used for training. Should be "rf" for RandomForest
        and "cb" for CatBoost.
    ckpt : str
        The filename inside 'checkpoint/' to load the model from. Should also contain the
        the filename extension.
    """
    Inferencer(model_type, ckpt).infer()


if __name__ == "__main__":
    fire.Fire()
