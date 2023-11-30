import pickle
from abc import ABCMeta, abstractmethod
from typing import Optional

import pandas as pd
from omegaconf import DictConfig


class BaseModel(metaclass=ABCMeta):
    """Represents an interface that any model used must implement."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.preprocessor = None
        self.model = None

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def eval(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, X_sample: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    def save_checkpoint(self, path: str) -> None:
        with open(path + self.cfg.training.checkpoint_name, "wb") as f:
            pickle.dump(self, f)

    @abstractmethod
    def log_fis_and_metrics(self, exp_id: str) -> None:
        raise NotImplementedError()
