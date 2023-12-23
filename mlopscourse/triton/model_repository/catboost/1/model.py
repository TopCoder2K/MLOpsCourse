import pickle
from typing import Any, List

import c_python_backend_utils as c_utils
import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        with open(f"/assets/{args['model_name']}.p", "rb") as f:
            self.model = pickle.load(f)

    @staticmethod
    def get_from_request_by_name(request: c_utils.InferenceRequest, name: str) -> Any:
        return pb_utils.get_input_tensor_by_name(request, name).as_numpy().tolist()[0]

    def execute(
        self, requests: List[c_utils.InferenceRequest]
    ) -> List[c_utils.InferenceResponse]:
        reqs = list()
        for request in requests:
            reqs.append(
                {
                    "season": TritonPythonModel.get_from_request_by_name(
                        request, "season"
                    )[0].decode(),
                    "weather": TritonPythonModel.get_from_request_by_name(
                        request, "weather"
                    )[0].decode(),
                    "month": TritonPythonModel.get_from_request_by_name(request, "month")[
                        0
                    ],
                    "hour": TritonPythonModel.get_from_request_by_name(request, "hour")[
                        0
                    ],
                    "holiday": TritonPythonModel.get_from_request_by_name(
                        request, "holiday"
                    )[0],
                    "weekday": TritonPythonModel.get_from_request_by_name(
                        request, "weekday"
                    )[0],
                    "workingday": TritonPythonModel.get_from_request_by_name(
                        request, "workingday"
                    )[0],
                    "temp": TritonPythonModel.get_from_request_by_name(request, "temp")[
                        0
                    ],
                    "feel_temp": TritonPythonModel.get_from_request_by_name(
                        request, "feel_temp"
                    )[0],
                    "humidity": TritonPythonModel.get_from_request_by_name(
                        request, "humidity"
                    )[0],
                    "windspeed": TritonPythonModel.get_from_request_by_name(
                        request, "windspeed"
                    )[0],
                }
            )
        preds = self.model(pd.DataFrame(reqs))

        responses = list()
        for pred in preds:
            responses.append(
                c_utils.InferenceResponse(
                    output_tensors=[
                        c_utils.Tensor("prediction", np.array(pred).reshape(1))
                    ]
                )
            )
        return responses
