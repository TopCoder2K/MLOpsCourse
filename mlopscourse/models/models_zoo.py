from typing import Optional, List, Dict
import argparse

from .base import BaseModel
from .random_forest import RandomForest
from .catboost import CatboostModel


def prepare_model(
    args: argparse.Namespace, numerical_features: List[str],
    categorical_features: List[str]
) -> BaseModel:
    if args.model == 'rf':
        return RandomForest(numerical_features, categorical_features)
    elif args.model == 'cb':
        return CatboostModel(numerical_features, categorical_features)
    else:
        assert False, f'Unknown model name: {args.model}'
