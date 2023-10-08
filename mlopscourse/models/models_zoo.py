from typing import List

from .base import BaseModel
from .catboost import CatboostModel
from .random_forest import RandomForest


def prepare_model(
    model_type: str,
    numerical_features: List[str],
    categorical_features: List[str],
) -> BaseModel:
    if model_type == "rf":
        return RandomForest(numerical_features, categorical_features)
    elif model_type == "cb":
        return CatboostModel(numerical_features, categorical_features)
    else:
        raise AssertionError(f"Unknown model name: {model_type}")
