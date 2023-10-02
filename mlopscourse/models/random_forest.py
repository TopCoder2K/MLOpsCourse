import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from .base import BaseModel


class RandomForest(BaseModel):
    """A basic Random Forest model from sklearn."""

    def __init__(
        self, numerical_features: List[str], categorical_features: List[str]
    ) -> None:
        super().__init__()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OrdinalEncoder(dtype=np.int64), categorical_features),
                ("num", "passthrough", numerical_features),
            ],
            sparse_threshold=1,
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        self.model = make_pipeline(
            self.preprocessor, RandomForestRegressor(random_state=0)
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> None:
        self.model.fit(X_train, y_train)
        if X_test is not None:
            assert y_test is not None, "For the evaluation, y_test must be provided!"
            self.eval(X_test, y_test)

    def eval(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
        print(f"Test R2 score: {self.model.score(X_test, y_test):.2f}")
        return pd.Series(self.model.predict(X_test), name="rf_preds")

    def __call__(self, X_sample: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_sample)

    def save_checkpoint(self, path: str) -> None:
        with open(path + "model_rf.p", "wb") as f:
            pickle.dump(self, f)
