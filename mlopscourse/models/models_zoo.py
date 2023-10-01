import argparse
from typing import List

from .base import BaseModel
from .catboost import CatboostModel
from .random_forest import RandomForest


def prepare_model(
    args: argparse.Namespace,
    numerical_features: List[str],
    categorical_features: List[str],
) -> BaseModel:
    if args.model == "rf":
        return RandomForest(numerical_features, categorical_features)
    elif args.model == "cb":
        return CatboostModel(numerical_features, categorical_features)
    else:
        raise AssertionError(f"Unknown model name: {args.model}")
