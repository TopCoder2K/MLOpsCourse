import argparse
import os

from mlopscourse.data.prepare_dataset import prepare_dataset
from mlopscourse.models.models_zoo import prepare_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training and evaluation parameters")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["rf", "cb"],
        help="Type of model used for training",
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        numerical_features,
        categorical_features,
    ) = prepare_dataset()
    model = prepare_model(args, numerical_features, categorical_features)

    print(f"Training the {args.model} model...")
    model.train(X_train, y_train)

    os.makedirs("checkpoints", exist_ok=True)
    model.save_checkpoint("checkpoints/")

    print(f"Finished!\nEvaluating the {args.model} model...")
    model.eval(X_test, y_test)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
