import requests
import numpy as np

input = np.array([[3, 2, 4, 0.2], [  4.7, 3, 1.3, 0.2 ]])

# seldon
inference_request = {"data": { "ndarray": input.tolist() }}
endpoint = "http://localhost:30907/api/v1.0/predictions"
response = requests.post(endpoint, json=inference_request, verify=False)
if response.status_code == 200:
    print(response.json())
else:
    print(response.status_code)
    print(response.content)

# mlserver
inference_request = {
    "inputs": [
        {
            "name": "predict",
            "datatype": "FP32",
            "shape": input.shape,
            "data": input.tolist(),
            "parameters": {
                "content_type": "np"
            }
        }
    ]
}
endpoint = "http://localhost:30908/v2/models/iris/infer"
response = requests.post(endpoint, json=inference_request, verify=False)
if response.status_code == 200:
    print(response.json())
else:
    print(response.status_code)
    print(response.content)

# fastapi
inference_request = {"data": input.tolist() }
endpoint = "http://localhost:30909/models/iris/predict"
response = requests.post(endpoint, json=inference_request, verify=False)
if response.status_code == 200:
    print(response.json())
else:
    print(response.status_code)
    print(response.content)