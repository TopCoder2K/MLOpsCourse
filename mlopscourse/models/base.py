from typing import Optional

import pandas as pd


class BaseModel:
    """Represents an interface that any model used must implement."""

    def __init__(self) -> None:
        self.preprocessor = None
        self.model = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> None:
        raise NotImplementedError()

    def eval(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
        raise NotImplementedError()

    def __call__(self, X_sample: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    def save_checkpoint(self, path: str) -> None:
        raise NotImplementedError()
