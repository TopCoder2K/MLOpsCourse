from typing import List, Optional

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm

from .base import BaseModel


class RandomForest(BaseModel):
    """A basic Random Forest model from sklearn."""

    def __init__(
        self,
        cfg: DictConfig,
        numerical_features: List[str],
        categorical_features: List[str],
    ) -> None:
        super().__init__(cfg)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OrdinalEncoder(dtype=np.int64), categorical_features),
                ("num", "passthrough", numerical_features),
            ],
            sparse_threshold=1,
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        self.model = make_pipeline(
            self.preprocessor, RandomForestRegressor(**cfg.model.hyperparams)
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

    def log_fis_and_metrics(
        self, exp_id: str, X_train: pd.DataFrame, y_train: pd.Series
    ) -> None:
        # Log the model's hyperparameters and the code version
        mlflow.log_params(self.cfg.model.hyperparams)
        mlflow.log_param("commit_id", self.cfg.logging.commit_id)
        # Log feature importances
        mlflow.log_metrics(
            {
                f"fi_of_{col_name}": self.model.named_steps[
                    "randomforestregressor"
                ].feature_importances_[i]
                for i, col_name in enumerate(X_train.columns)
            }
        )
        # Log R2 and RMSE metrics
        for i in tqdm(range(0, self.cfg.model.hyperparams.n_estimators)):
            model_i = make_pipeline(
                self.preprocessor, RandomForestRegressor(**self.cfg.model.hyperparams)
            )
            model_i.named_steps["randomforestregressor"].n_estimators = i + 1
            model_i.fit(X_train, y_train)
            y_pred = model_i.predict(X_train)
            mlflow.log_metrics(
                {
                    "R2_metric": r2_score(y_train, y_pred),
                    "RMSE_metric": mean_squared_error(y_train, y_pred, squared=False),
                },
                step=i,
            )
