from typing import List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from omegaconf import DictConfig
from sklearn.metrics import r2_score

from .base import BaseModel


class CatboostModel(BaseModel):
    """The Yandex's CatBoost."""

    def __init__(
        self,
        cfg: DictConfig,
        numerical_features: List[str],
        categorical_features: List[str],
    ) -> None:
        super().__init__(cfg)

        self.model = CatBoostRegressor(**cfg.model.hyperparams)
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> None:
        train_data = Pool(
            data=X_train,
            label=y_train,
            cat_features=self.categorical_features,
            feature_names=list(X_train.columns),
        )
        if X_test is not None:
            assert y_test is not None, "For the evaluation, y_test must be provided!"
            test_data = Pool(
                data=X_test,
                label=y_test,
                cat_features=self.categorical_features,
                feature_names=list(X_test.columns),
            )
            self.model.fit(train_data, eval_set=test_data, use_best_model=True)
        else:
            self.model.fit(train_data)

    def eval(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
        test_data = Pool(
            data=X_test,
            label=y_test,
            cat_features=self.categorical_features,
            feature_names=list(X_test.columns),
        )
        preds = self.model.predict(test_data)
        print("CatBoost R2: {:.2f}".format(r2_score(y_test, preds)))
        return pd.Series(preds, name="cb_preds")

    def __call__(self, X_sample: pd.DataFrame) -> np.ndarray:
        sample_data = Pool(
            data=X_sample,
            label=None,
            cat_features=self.categorical_features,
            feature_names=list(X_sample.columns),
        )
        return self.model.predict(sample_data)
