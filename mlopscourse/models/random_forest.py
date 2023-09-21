from typing import Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from .base import BaseModel


class RandomForest(BaseModel):
    """
    A basic Random Forest model from sklearn.
    """
    def __init__(
        self, numerical_features, categorical_features,
        hyperparams: Optional[dict] = None
    ) -> None:
        super().__init__(hyperparams)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OrdinalEncoder(dtype=np.int64), categorical_features),
                ('num', 'passthrough', numerical_features),
            ],
            sparse_threshold=1,
            verbose_feature_names_out=False,
        ).set_output(transform='pandas')
        self.model = make_pipeline(
            self.preprocessor,
            RandomForestRegressor(
                random_state=0
            ),
        )

    # TODO: types
    def train(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def eval(self, X_test, y_test) -> None:
        print(f"Test R2 score: {self.model.score(X_test, y_test):.2f}")

    # TODO:
    def __call__(self, X_sample):
        pass
