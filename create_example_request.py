import json

import fire
import pandas as pd


def create_example_request() -> None:
    example_df = pd.DataFrame(
        [
            {
                "season": "spring",
                "month": 1,
                "hour": 0,
                "holiday": False,
                "weekday": 6,
                "workingday": False,
                "weather": "clear",
                "temp": 9.84,
                "feel_temp": 14.395,
                "humidity": 0.81,
                "windspeed": 0.0,
            }
        ]
    )  # This is the first row of the training split
    with open("example_request.json", "w") as f:
        json.dump({"dataframe_split": example_df.to_dict(orient="split")}, f)


if __name__ == "__main__":
    fire.Fire()
