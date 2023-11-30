from typing import List, Optional

import mlflow
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

    def log_fis_and_metrics(self, exp_id: str, col_names: List[str]) -> None:
        with mlflow.start_run(
            experiment_id=exp_id, run_name=f"training-{self.cfg.model.name}"
        ):
            # Log the model's hyperparameters and the code version
            mlflow.log_params(self.cfg.model.hyperparams)
            mlflow.log_param("commit_id", self.cfg.logging.commit_id)
            # Log feature importances
            mlflow.log_metrics(
                {
                    f"fi_of_{col_name}": self.model.feature_importances_[i]
                    for i, col_name in enumerate(col_names)
                }
            )
            r2_scores = self.model.evals_result_["learn"]["R2"]
            rmse_scores = self.model.evals_result_["learn"]["RMSE"]
            assert len(r2_scores) == len(rmse_scores), "Something wrong with metrics!"
            for i in range(len(r2_scores)):
                mlflow.log_metrics(
                    {"R2_metric": r2_scores[i], "RMSE_loss": rmse_scores[i]},
                    step=i * self.cfg.model.hyperparams.metric_period,
                )
