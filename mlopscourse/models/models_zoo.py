from typing import Optional
import argparse

from .random_forest import RandomForest
from .base import BaseModel


def prepare_model(
    args: argparse.Namespace, numerical_features, categorical_features,
    hyperparams: Optional[dict] = None
) -> BaseModel:
    if args.model == 'rf':
        return RandomForest(
            numerical_features, categorical_features, hyperparams
        )
    else:
        assert False, f'Unknown model name: {args.model}'
