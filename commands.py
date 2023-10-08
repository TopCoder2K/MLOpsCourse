import fire

from mlopscourse.infer import infer
from mlopscourse.train import train


def outer_train(model_type: str) -> None:
    """
    Trains the chosen model on the train split of the dataset and saves the checkpoint.

    Parameters
    ----------
    model_type : str
        The type of model for training. Should be "rf" for RandomForest and "cb"
        for CatBoost.
    """
    train(model_type)


def outer_infer(model_type: str, ckpt: str) -> None:
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
    infer(model_type, ckpt)


if __name__ == "__main__":
    fire.Fire()
