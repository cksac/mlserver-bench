import requests

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

response = requests.post("http://localhost:9091/v2/models/model-a/infer", json=inference_request)
if response.status_code == 200:
    print(response.json())
else:
    print(response.status_code, response.content)