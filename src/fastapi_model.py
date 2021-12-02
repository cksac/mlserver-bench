from iris_classifier import IrisClassifier
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class Input(BaseModel):
    data: list

class Output(BaseModel):
    data: list

model = IrisClassifier()

@app.post("/models/iris/predict")
async def predict(input: Input) -> Output:
    pred = model.predict(np.array(input.data))
    return Output(
        data=pred.tolist()
    )