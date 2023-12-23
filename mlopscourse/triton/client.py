import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


def test_catboost_with_triton():
    example = {
        "season": "spring".encode("utf-8"),
        "month": 1,
        "hour": 0,
        "holiday": 0,
        "weekday": 6,
        "workingday": 0,
        "weather": "clear".encode("utf-8"),
        "temp": 9.84,
        "feel_temp": 14.395,
        "humidity": 0.81,
        "windspeed": 0.0,
    }  # This is the first row of the training split
    input_example = list()
    for k, v in example.items():
        if k in ["temp", "feel_temp", "humidity", "windspeed"]:
            v = np.array(
                [
                    v,
                ],
                dtype=np.float32,
            ).reshape(-1, 1)
        elif k in ["month", "hour", "holiday", "weekday", "workingday"]:
            v = np.array(
                [
                    v,
                ],
                dtype=np.int32,
            ).reshape(-1, 1)
        else:
            v = np.array(
                [
                    v,
                ]
            ).reshape(-1, 1)
        input_example.append(
            InferInput(
                name=k, shape=[1, 1], datatype=np_to_triton_dtype(v.dtype)
            ).set_data_from_numpy(v)
        )

    client = InferenceServerClient(url="localhost:8000")
    result = client.infer(
        "catboost",
        input_example,
        outputs=[
            InferRequestedOutput("prediction"),
        ],
    )
    expected_pred = 31.22848957148021  # Is taken from the mlflow inference result
    assert (
        expected_pred == result.as_numpy("prediction")[0]
    ), "Something is wrong with the inference :(("
    print("Predicted:", result.as_numpy("prediction")[0])
    print("The test is passed!")


if __name__ == "__main__":
    test_catboost_with_triton()
