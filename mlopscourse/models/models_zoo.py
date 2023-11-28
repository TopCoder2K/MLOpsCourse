from typing import List

from omegaconf import DictConfig

from .base import BaseModel
from .catboost import CatboostModel
from .random_forest import RandomForest


def prepare_model(
    cfg: DictConfig,
    numerical_features: List[str],
    categorical_features: List[str],
) -> BaseModel:
    if cfg.model.name == "rf":
        return RandomForest(cfg, numerical_features, categorical_features)
    elif cfg.model.name == "cb":
        return CatboostModel(cfg, numerical_features, categorical_features)
    else:
        raise AssertionError(f"Unknown model name: {cfg.model.name}")
