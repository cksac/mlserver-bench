from os import name
from mlserver import MLModel
from mlserver.settings import ModelSettings
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
import pandas as pd
import numpy as np
import os
import pickle
import time

class Model(MLModel):
    def __init__(self, settings: ModelSettings):
        super().__init__(settings)
        self.model_dir = settings.parameters.extra.get("model_dir", ".")        
        self.cols = ["COL_1", "COL_2", "COL_3", "COL_4", "COL_5", "COL_6", "COL_7", "COL_8", "COL_9"]

    async def load(self) -> bool:        
        # simulate load sub_model_1, 2
        with open(os.path.join(self.model_dir, "dummy_model.pkl"), "rb") as f:
            self._dummy_model = pickle.load(f)

        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        data = self._extract_input(payload)
        input = self._transform(data)

        self._predict_sub_model_1(input)
        self._predict_sub_model_2(input)

        # introduct some predict sync computation
        # simulate sub model predict call, 100ms
        time.sleep(0.1)

        response = self._build_response(payload, input)
        return response

    def _predict_sub_model_1(self, input: pd.DataFrame):
        input['PRED_1'] = np.random.uniform(1, 6, input.shape[0])

    def _predict_sub_model_2(self, input: pd.DataFrame):
        input['PRED_2'] = np.random.uniform(1, 6, input.shape[0])

    def _extract_input(self, payload):
        request_input = payload.inputs[0]
        data = self.decode(request_input)
        return data

    def _transform(self, data) -> pd.DataFrame:
        input = pd.DataFrame([data], columns=self.cols)
        return input

    def _build_response(self, payload, input) -> InferenceResponse:
        data = input[["PRED_1", "PRED_2"]].to_numpy()[0].tolist()
        return InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            id=payload.id,
            outputs=[
                ResponseOutput(
                    name="predict",
                    shape=[2],
                    datatype="FP32",
                    data=data
                )
            ]
        )


if __name__ == "__main__":
    import asyncio

    loop = asyncio.get_event_loop()

    model = Model(ModelSettings(
        name="model"
    ))

    inference_request = {
        "inputs": [
            {
                "name": "predict",
                "datatype": "BYTES",
                "shape": [9],
                "data": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
                "parameters": {
                    "content_type": "str"
                }
            }
        ]
    }
    request = InferenceRequest.parse_obj(inference_request)
    response =  loop.run_until_complete(model.predict(request))
    print(response)
