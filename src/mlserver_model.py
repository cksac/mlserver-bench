from mlserver import MLModel, types
from mlserver.settings import ModelSettings
from typing import Iterable, Dict, Optional, Union, List
import numpy as np
from iris_classifier import IrisClassifier

class MLServerModel(MLModel):
    def __init__(self, settings: ModelSettings):
        super().__init__(settings)
        self.iris_classifier = IrisClassifier()
        self.ready = False

    async def load(self) -> bool:
        self.iris_classifier.load()
        self.ready = True
        return self.ready

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        if not self.ready:
            await self.load()

        input = payload.inputs[0].data
        pred = self.iris_classifier.predict(input)
        return types.InferenceResponse(
            model_name=self.name,
            id=payload.id,
            outputs=[
                types.ResponseOutput(
                    name=payload.inputs[0].name,
                    shape=pred.shape,
                    datatype="FP32",
                    data=pred.tolist(),
                )
            ],
        )


if __name__ == "__main__":
    import asyncio
    loop = asyncio.get_event_loop()
    
    input = np.array([[3, 2, 4, 0.2], [  4.7, 3, 1.3, 0.2 ]])
    inference_request = {
        "inputs": [
            {
                "name": "predict",
                "datatype": "FP32",
                "shape": input.shape,
                "data": input,
                "parameters": {
                    "content_type": "np"
                }
            }
        ]
    }
    request = types.InferenceRequest.parse_obj(inference_request)
    model = MLServerModel(ModelSettings())
    output = loop.run_until_complete(model.predict(request))
    print(output)
