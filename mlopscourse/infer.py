import argparse
import os
import pickle

from mlopscourse.data.prepare_dataset import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training and evaluation parameters")

    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "cb"],
        help="Type of model used for training",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="model_rf.p",
        help="The filename inside 'checkpoint/' to load the model from",
    )

    args = parser.parse_args()
    return args


def infer(args: argparse.Namespace):
    (
        _,
        _,
        X_test,
        y_test,
        _,
        _,
    ) = prepare_dataset(print_info=False)

    with open(f"checkpoints/{args.ckpt}", "rb") as f:
        model = pickle.load(f)
    print(f"Evaluating the {args.model} model...")
    y_preds = model.eval(X_test, y_test)

    os.makedirs("predictions", exist_ok=True)
    y_preds.to_csv(f"predictions/model_{args.model}_preds.csv")


if __name__ == "__main__":
    arguments = parse_args()
    infer(arguments)
