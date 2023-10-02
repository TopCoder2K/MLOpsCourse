import argparse
import os

from data.prepare_dataset import prepare_dataset
from models.models_zoo import prepare_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training and evaluation parameters")

    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "cb"],
        help="Type of model used for training",
    )

    args = parser.parse_args()
    return args


def train(args: argparse.Namespace):
    (
        X_train,
        y_train,
        _,
        _,
        numerical_features,
        categorical_features,
    ) = prepare_dataset()
    model = prepare_model(args, numerical_features, categorical_features)

    print(f"Training the {args.model} model...")
    model.train(X_train, y_train)

    os.makedirs("checkpoints", exist_ok=True)
    model.save_checkpoint("checkpoints/")


if __name__ == "__main__":
    arguments = parse_args()
    train(arguments)
